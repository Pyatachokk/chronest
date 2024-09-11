class DateError(ValueError):
    """Raise when some dates are incorrect"""

    def __init__(self, *args):
        super().__init__(*args)

class DataError(ValueError):
    """Raise when incoming data has incorrect properties"""

    def __init__(self, *args):
        super().__init__(*args) 