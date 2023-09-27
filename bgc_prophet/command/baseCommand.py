

class baseCommand:
    """
    Base command class for BGC-Prophet
    """
    def add_arguments(self, parser):
        """
        Add arguments to the parser
        """
        raise NotImplementedError

    def handle(self, args):
        """
        Handle the command
        """
        raise NotImplementedError
    