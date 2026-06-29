class BaseTool:
    name: str = "base"
    description: str = ""

    def run(self, args: dict) -> str:
        raise NotImplementedError
