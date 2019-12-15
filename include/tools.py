class LogManager:
    def __init__(self,log_file_name,stdout=False):
        self.log_file_name = log_file_name
        self.option = stdout

    def log_write(self,sentence):
        f = open(self.log_file_name,"a")
        f.write(sentence + '\n')
        f.close()
        if(self.option):
            print(sentence)

