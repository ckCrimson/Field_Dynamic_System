from abc import ABC

from fds.affecting.affecting_systems_framework.is_affecting import IsAffecting


class IsCorrelationAffected(IsAffecting, ABC):
    def __init__(self,**params):
        super().__init__(**params)