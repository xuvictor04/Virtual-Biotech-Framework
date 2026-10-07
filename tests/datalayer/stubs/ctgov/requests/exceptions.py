"""The exception names the upstream clinicaltrials tools catch."""


class RequestException(Exception):
    pass


class Timeout(RequestException):
    pass
