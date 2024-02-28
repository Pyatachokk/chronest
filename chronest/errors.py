class DateError(ValueError):
    """Raise when some dates are incorrect"""

    def __init__(self, *args):
        super().__init__(*args)
