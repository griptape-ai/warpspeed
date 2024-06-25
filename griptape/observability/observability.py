from types import TracebackType
from typing import Any, Optional
from attrs import define, field
from griptape.common import Observable
from griptape.drivers import BaseObservabilityDriver, NoOpObservabilityDriver

_no_op_observability_driver = NoOpObservabilityDriver()
_global_observability_driver: Optional[BaseObservabilityDriver] = None


@define
class Observability:
    observability_driver: BaseObservabilityDriver = field(kw_only=True)

    @staticmethod
    def get_no_op_driver() -> NoOpObservabilityDriver:
        global _no_op_observability_driver
        if _no_op_observability_driver is None:
            _no_op_observability_driver = NoOpObservabilityDriver()
        return _no_op_observability_driver

    @staticmethod
    def get_global_driver() -> Optional[BaseObservabilityDriver]:
        global _global_observability_driver
        return _global_observability_driver

    @staticmethod
    def set_global_driver(driver: Optional[BaseObservabilityDriver]):
        global _global_observability_driver
        _global_observability_driver = driver

    @staticmethod
    def observe(call: Observable.Call) -> Any:
        driver = Observability.get_global_driver() or Observability.get_no_op_driver()
        return driver.observe(call)

    def __enter__(self):
        if Observability.get_global_driver() is not None:
            raise ValueError("Observability driver already set.")
        Observability.set_global_driver(self.observability_driver)
        self.observability_driver.__enter__()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> bool:
        Observability.set_global_driver(None)
        self.observability_driver.__exit__(exc_type, exc_value, exc_traceback)
        return False
