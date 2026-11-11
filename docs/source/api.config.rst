Configuration
------------------------------

The global configuration object is available as ``flodym.config``.
Options can be changed by assignment, e.g.
``flodym.config.error_behaviors.unset_array_used = "warn"``,
or temporarily within a ``with flodym.config.override(...):`` block.

The configuration controls

- which checks are performed automatically (:py:class:`flodym.Checks`), e.g. the mass
  balance check after each stock or process computation (off by default),
- what happens if a check fails (:py:class:`flodym.ErrorBehaviors`): raise an error, log a
  warning or info message, or ignore it. Besides mass balance and flow checks, this concerns the
  automatic computation of processes (:py:meth:`flodym.Process.compute`): underdetermined
  processes, invalid shares or dimension splitters, inconsistent values of overdetermined
  processes, and unused dimension splitters, as well as operations which are inconsistent with
  the extensivity of arrays (see :py:attr:`flodym.FlodymArray.extensive`),
- the tolerance of mass balance and other checks (``relative_tolerance`` and
  ``absolute_tolerance`` of :py:class:`flodym.Config`).

.. autopydantic_model:: flodym::Config

.. autopydantic_model:: flodym::Checks

.. autopydantic_model:: flodym::ErrorBehaviors

.. autoclass:: flodym::ErrorBehavior
   :members:
   :undoc-members:

Validation errors
------------------------------

Invalid arguments of flodym classes, e.g. a misspelled argument name or a value of the wrong
type, raise a :py:class:`flodym.FlodymValidationError`. The same applies to invalid values
assigned to configuration options.

.. autoexception:: flodym.FlodymValidationError
   :members:

Data errors
------------------------------

Data that do not fit the flodym object they are read for raise a
:py:class:`flodym.FlodymDataError`, e.g. a DataFrame whose columns cannot be assigned to the
dimensions of a parameter, items that are not in a dimension, or values that are not numbers.
The message first names where the data came from, e.g. the parameter and the file, and then
describes the problem.

.. autoexception:: flodym.FlodymDataError
