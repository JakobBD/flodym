from typing import List, Dict
from pydantic import BaseModel as PydanticBaseModel, model_validator
import logging
from functools import reduce
import numpy as np

from .flodym_arrays import Flow, FlodymArray, Parameter
from .stocks import Stock, StockDrivenDSM, InflowDrivenDSM, SimpleFlowDrivenStock
from .mfa_definition import ProcessDefinition
from .dimensions import DimensionSet


class UnderdeterminedError(Exception):
    """Exception raised when a process is underdetermined."""

    def __init__(self, process: "Process", stage: str, sides: List[str]):
        message= f"Cannot compute process '{process.name}' with ID {process.id}, as it is underdetermined. \n"
        if stage == "total":
            message += "Failed to compute total flow of process from given in/outflows.\n"
            for side in sides:
                message += f"Tried to compute from {side}flows, but failed. Reasons:\n"
                if not process.known_flows(side):
                    message += f"- no flow from this side has an absolute value given"
                for linked_process in process.neither_known(side):
                    message += f"- for flow from/to process {linked_process}, neither share nor absolute value is given.\n"
        elif stage == "flows":
            for side in sides:
                message += f"Failed to compute {side}flows from total; For more than one {side}flow, neither share nor absolute value was given.\n"
                message += f"Please provide either share or absolute value for all but one of the flows from/to the processes "
                message += ", ".join(process.neither_known(side)) + "."
        super().__init__(message)
        self.message = message

class Process(PydanticBaseModel):
    """Processes serve as nodes for the MFA system layout definition.
    Flows are defined between two processes. Stocks are connected to a process.
    Processes do not contain values themselves.

    Processes get an ID by the order they are defined in the :py:attribute::`MFASystem.definition`.
    The process with ID 0 necessarily contains everything outside the system boundary.
    It has to be named 'sysenv'.
    """

    name: str
    """Name of the process."""
    id: int
    """ID of the process."""
    _inflows: Dict[str, Flow] = {}
    """Inflows to the process, keyed by linked process name."""
    _outflows: Dict[str, Flow] = {}
    """Outflows from the process, keyed by linked process name."""
    _stock: Stock = None
    dimension_splitter: "FlodymArray" = None
    _inflow_shares: Dict[str, "FlodymArray"] = {}
    _outflow_shares: Dict[str, "FlodymArray"] = {}
    _total: FlodymArray = None

    @model_validator(mode="after")
    def check_id0(self):
        """Ensure that the process with ID 0 is named 'sysenv'."""
        if self.id == 0 and self.name != "sysenv":
            raise ValueError(
                "The process with ID 0 must be named 'sysenv', as it contains everything outside the system boundary."
            )
        return self

    @property
    def inflows(self) -> Dict[str, Flow]:
        """Inflows to the process."""
        return self._inflows

    @property
    def outflows(self) -> Dict[str, Flow]:
        """Outflows from the process."""
        return self._outflows

    def add_inflow(self, flow: Flow) -> None:
        """Add an inflow flow to the process."""
        self._inflows[flow.from_process.name] = flow

    def add_outflow(self, flow: Flow) -> None:
        """Add an outflow flow from the process."""
        self._outflows[flow.to_process.name] = flow

    def set_stock(self, stock: Stock) -> None:
        """Set the stock associated with this process."""
        self._stock = stock

    def set_inflow_share(self, from_process: str, share: FlodymArray):
        if from_process not in self._inflows:
            raise ValueError(f"No inflow from process '{from_process}' found.")
        self._inflow_shares[from_process] = share

    def set_outflow_share(self, to_process: str, share: FlodymArray):
        if to_process not in self._outflows:
            raise ValueError(f"No outflow to process '{to_process}' found.")
        self._outflow_shares[to_process] = share

    def is_computed(self) -> bool:
        """Check if the process has been computed."""
        if self.unknown_flows("in") or self.unknown_flows("out"):
            return False
        if self._stock is not None and not self._stock.is_computed():
            return False
        return True

    def compute(self, underdetermined_behavior: str = "error", recursive: bool=False) -> None:
        if self.is_computed():
            logging.debug(f"Process {self.name} with ID {self.id} is already computed.")
            return
        if self.id == 0:
            logging.debug(f"Process {self.name} with ID {self.id} is the system boundary and cannot be computed.")
            return
        try:
            self.try_compute()
            if recursive:
                for flow in self.inflows.values():
                    flow.from_process.compute(underdetermined_behavior=underdetermined_behavior, recursive=True)
                for flow in self.outflows.values():
                    flow.to_process.compute(underdetermined_behavior=underdetermined_behavior, recursive=True)
        except UnderdeterminedError as e:
            self.handle_underdetermined(error=e, underdetermined_behavior=underdetermined_behavior)

    def try_compute(self):
        if self._stock is None:
            self.compute_no_stock()
        elif isinstance(self._stock, StockDrivenDSM):
            self.compute_stock_driven()
        elif isinstance(self._stock, InflowDrivenDSM):
            self.compute_inflow_driven()
        elif isinstance(self._stock, SimpleFlowDrivenStock):
            self.compute_simple_flow_driven()
        if self.unknown_flows("in") or self.unknown_flows("out"):
            raise ValueError(
                f"In Process {self.name}: After computation, there are still unknown flows. "
                "This indicates an internal flodym error."
            )

    def compute_no_stock(self):
        self.compute_total()
        self.apply_dimension_splitter()
        self.compute_flows()

    def compute_stock_driven(self):

        self._stock.compute()

        self._total = self._stock.outflow
        self.apply_dimension_splitter(sides=["out"])
        self.compute_flows(sides=["out"])

        self._total = self._stock.inflow
        self.apply_dimension_splitter(sides=["in"])
        self.compute_flows(sides=["in"])

    def compute_inflow_driven(self):
        if self._stock.inflow.is_set:
            self._stock.compute()
            self._total = self._stock.inflow
            self.compute_flows(sides=["in"])
        else:
            self.compute_total(try_sides=["in"])
            self._stock.inflow[...] = self._total
            self._stock.compute()

        self._total = self._stock.outflow
        self.apply_dimension_splitter(sides=["out"])
        self.compute_flows(sides=["out"])

    def compute_simple_flow_driven(self):
        if self.inflows:
            self.compute_total(try_sides=["in"])
            if self._stock.inflow.dims - self._total.dims:
                names = ", ".join((self._stock.inflow.dims - self._total.dims).names)
                raise ValueError(
                    f"In Process {self.name}: Stock inflow has dimensions {names} not contained in the "
                    "dimensions of the summed inflow Flows. Consider using less dimensions for the "
                    "stock or a preceding process with a dimension_splitter."
                )
            self._stock.inflow[...] = self._total

        if self.outflows:
            self.compute_total(try_sides=["out"])
            if self._stock.outflow.dims - self._total.dims:
                names = ", ".join((self._stock.outflow.dims - self._total.dims).names)
                raise ValueError(
                    f"In Process {self.name}: Stock outflow has dimensions {names} not contained in the "
                    "dimensions of the summed outflow Flows. Consider using less dimensions for the "
                    "stock or a neighboring process with a dimension_splitter."
                )
            self._stock.outflow[...] = self._total

        self._stock.compute()

    def handle_underdetermined(self, error: UnderdeterminedError, underdetermined_behavior: str = "error") -> bool:
        if underdetermined_behavior == "error":
            raise error
        elif underdetermined_behavior == "warn":
            logging.warning(error.message)
        elif underdetermined_behavior == "info":
            logging.info(f"Process {self.name} is underdetermined. Skip computation.")
        elif underdetermined_behavior == "ignore":
            return
        else:
            raise ValueError(f"Unknown behavior: {underdetermined_behavior}")

    def flows(self, direction: str):
        if direction == "in":
            return self._inflows
        elif direction == "out":
            return self._outflows
        else:
            raise ValueError("Direction must be 'in' or 'out'.")

    def shares(self, side: str):
        if side == "in":
            return self._inflow_shares
        elif side == "out":
            return self._outflow_shares
        else:
            raise ValueError("Direction must be 'in' or 'out'.")

    def neither_known(self, side: str) -> List[str]:
        return [
            name for name, flow in self.flows(side).items() if not flow.is_set
            and name not in self.shares(side)
        ]

    def known_flows(self, side: str) -> Dict[str, Flow]:
        return {
            name: flow for name, flow in self.flows(side).items() if flow.is_set
        }

    def unknown_flows(self, side: str) -> Dict[str, Flow]:
        return {
            name: flow for name, flow in self.flows(side).items() if not flow.is_set
        }

    def both_known(self, side: str) -> List[str]:
        return [
            name for name, flow in self.flows(side).items() if flow.is_set
            and name in self.shares(side)
        ]

    # def sides(self, direction: str) -> List[str]:
    #     if direction == "forward":
    #         return ["in", "out"]
    #     elif direction == "backward":
    #         return ["out", "in"]
    #     else:
    #         raise ValueError("Direction must be 'forward' or 'backward'.")

    def can_compute_total(self, from_side: str) -> bool:
        """Check if the process can compute the total flow through the process in the given direction."""
        return len(self.neither_known(from_side)) == 0 and len(self.known_flows(from_side)) >= 1

    def can_compute_flows(self, sides: List[str] = ["in", "out"]) -> bool:
        """Check if the process can compute the flows in the given direction."""
        return all(len(self.neither_known(side)) <= 1 for side in sides)

    def compute_total(self, try_sides: List[str] = ["in", "out"]):
        """Compute the total flow based on the inflows or outflows."""
        side = None
        for s in try_sides:
            if self.can_compute_total(s):
                side = s
                break
        if side is None:
            raise UnderdeterminedError(process=self, stage="total", sides=try_sides)

        if len(self.known_flows(side)) == len(self.flows(side)):
            # all flows are known
            self._total = sum(self.flows(side).values())
        elif self.both_known(side):
            name = self.both_known(side)[0]
            excess_dims = self.shares(side)[name].dims - self.flows(side)[name].dims
            if excess_dims:
                names = ", ".join(excess_dims.names)
                raise ValueError(
                    f"In Process {self.name}: Share of flow to/from process {name} has dimensions "
                    f"{names} not contained in the flow's dimensions."
                )
            self._total = self.flows(side)[name] / self.shares(side)[name]
        else:
            sum_known = sum(self.known_flows(side).values())
            shares_unknown = {name: self.shares(side)[name] for name in self.unknown_flows(side)}
            dims_unknown = reduce(
                lambda x, y: x | y.dims, shares_unknown.values(), DimensionSet(dim_list=[])
            )
            if shares_unknown - sum_known.dims:
                share_names = ", ".join(
                    [name for name, share in shares_unknown.items() if share.dims - sum_known.dims]
                )
                excess_names = ", ".join((dims_unknown-sum_known.dims).names)
                raise ValueError(
                    f"In Process {self.name}: Error when trying to infer total flow from known "
                    f"flows and shares of unknown flows: Shares of flows to/from process(es) "
                    f"{share_names} has dimensions " f"{excess_names} not contained in all the "
                    f"known flows from/to processes {', '.join(self.known_flows(side))}"
                )
            sum_shares_unknown = sum([s.cast_to(dims_unknown) for s in shares_unknown.values()])
            self._total = sum_known / (1 - sum_shares_unknown)

    def apply_dimension_splitter(self, sides: list[str] = ["in", "out"]):
        """Apply the dimension splitter to the total flow."""

        dims_unknown = DimensionSet(dim_list=[])
        for side in sides:
            for flow in self.unknown_flows(side).values():
                dims_unknown |= flow.dims
        missing_dims = dims_unknown - self._total.dims

        if not missing_dims:
            return

        if self._dimension_splitter is None:
            raise ValueError(
                f"Process {self.name} has missing dimensions {missing_dims.names} for unknown flows, but no dimension splitter is set."
            )

        splitter_dims = self._dimension_splitter.dims
        if missing_dims - splitter_dims:
            raise ValueError(
                f"Dimension splitter of process {self.name} does not cover all dimensions {missing_dims.names} needed to determine unknown flows from total flow."
            )

        common_dims = splitter_dims & self._total.dims
        splitter_sum = self._dimension_splitter.sum_to(common_dims.letters)
        tolerance = 1e-6 # TODO
        if np.max(np.abs((splitter_sum.values - 1))) > tolerance:
            raise ValueError(
                f"Dimension splitter of process {self.name} does not sum to 1 if summed to common dimensions with process total {common_dims.names}."
            )

        self._total *= self._dimension_splitter

    def compute_flows(self, sides: List[str] = ["in", "out"]):
        """Compute the flows based on the total flow."""

        if not self.can_compute_flows(sides):
            raise UnderdeterminedError(process=self, stage="flow", sides=sides)

        for side in sides:
            # saving in temp flow avoids reducing dimensions.
            # This may be important for inferring the last flow by subtracting all known from the total,
            # because the dimension set intersection is used for that subtraction.
            temp_flows = {}
            for share_name, share in self.shares(side).items():
                if share_name in self.unknown_flows(side):
                    if share.dims - self._total.dims:
                        names = ", ".join((share.dims - self._total.dims).names)
                        raise ValueError(
                            f"In Process {self.name}: Share of flow to/from process {share_name} "
                            f"has dimensions {names} not contained in the processes' total "
                            f"dimensions {', '.join(self._total.dims.names)}. If you wish to "
                            "perform this dimensional split, consider outsourcing it to its own "
                            "process with a dimension_splitter."
                        )
                    temp_flows[share_name] = self._total * share
            # calculate the last by sum, if necessary
            if len(self.unknown_flows(side)) - len(temp_flows) == 1:
                unknown_name = [n for n in self.unknown_flows(side) if n not in temp_flows][0]
                unknown_flow = self.unknown_flows(side)[unknown_name]
                sum_known = sum(self.known_flows(side).values()) + sum(temp_flows.values())
                if isinstance(sum_known, FlodymArray) and unknown_flow.dims - sum_known.dims:
                    known_names = ", ".join(self.known_flows(side))
                    missing_names = ", ".join((unknown_flow.dims - sum_known.dims).names)
                    raise ValueError(
                        f"In Process {self.name}: One of the flows with set values ({known_names}) "
                        f"is missing dimensions ({missing_names}), which are needed to compute "
                        f"the unknown flow from/to process {unknown_name} by subtracting known "
                        "flows from the total."
                    )
                unknown_flow[...] = self._total - sum_known
            # reduce dimensions and transfer to actual flows
            for flow_name, flow in temp_flows.items():
                self.flows(side)[flow_name][...] = flow

def make_processes(definitions: List[str|ProcessDefinition]) -> dict[str, Process]:
    """Create a dictionary of processes from a list of process names."""
    processes = {}
    for definition in definitions:
        if isinstance(definition, str):
            name = definition
            id = len(processes)
        elif isinstance(definition, ProcessDefinition):
            if definition.id != len(processes):
                raise ValueError(f"Processes must be defined with consecutive IDs starting from 0, but found {definition.id} in {len(processes)}'th definition.")
            name = definition.name
            id = definition.id
        processes[name] = Process(name=name, id=id)
    return processes


def set_process_parameters(processes: Dict[str, Process], definitions: List[ProcessDefinition], parameters: Dict[str, Parameter]):
    for definition in definitions:
        if isinstance(definition, str):
            continue

        if definition.name not in processes:
            raise ValueError(f"Process {definition.name} is not defined in the processes dictionary.")
        process = processes[definition.name]

        if definition.inflow_shares:
            for from_process, parameter_name in definition.inflow_shares.items():
                if parameter_name not in parameters:
                    raise ValueError(f"Parameter {parameter_name} given in definition of process {definition.name} is not defined.")
                process.set_inflow_share(from_process, parameters[parameter_name])

        if definition.outflow_shares:
            for to_process, parameter_name in definition.outflow_shares.items():
                if parameter_name not in parameters:
                    raise ValueError(f"Parameter {parameter_name} given in definition of process {definition.name} is not defined.")
                process.set_outflow_share(to_process, parameters[parameter_name])

        if definition.dimension_splitter:
            if definition.dimension_splitter not in parameters:
                raise ValueError(f"Dimension splitter {definition.dimension_splitter} given in definition of process {definition.name} is not defined.")
            process.dimension_splitter = parameters[definition.dimension_splitter]

# conditions for well-determinedness: [no underdeterminedness]
# - (n_i + n_o) - 1 knowns: [or more]
#   - n_i + n_o + 1 unknowns (a_i, a_o, total)
#   - two conditions: sum a_i = 1, sum a_o = 1
# - total number of i/o with neither share nor value must be one [not exceed one]
# - at least one absolute value

# dimensions:
# - before total: all prm dims must be contained in the flow
# - total: dimension splitter may be applied
#   - consistency check: when summed to dim intersection with total, must be one
# - after total: all prm dims must be contained in the total
