class Pipeline:
    """Base class for image processing pipeline"""
    def __init__(self, path_to_image: str, is_ref: bool):
        self.path = path_to_image
        self.state = "raw"
        self.ref = is_ref

    def __str__(self) -> str:
        return "{0}: [state = {1}] [ref = {2}]".format(self.path, self.state, self.ref)
    
    def init_log(path_to_log) -> None:
        try:
            with open(path_to_log,"w") as f:
                f.write("Log file for image processing pipeline\n")
        except:
            raise Exception("Log creation failed")
        return None
    
    def clear_log(path_to_log) -> None:
        try:
            with open(path_to_log,"w") as f:
                f.write("")
        except:
            raise Exception("Log clear failed")
        return None
    
    def to_log(self,path_to_log) -> None:
        try:
            with open(path_to_log,"a") as f:
                f.write("".join(str(self),"\n"))
        except:
            raise Exception("Log failed")
        return None
    
