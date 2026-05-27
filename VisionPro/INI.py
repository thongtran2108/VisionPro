import re

class Ini:
    def __init__(self, file_path: str):
        self.path = file_path

    def find(self, data: str):
        try:
            with open(self.path, 'r') as file:
                content = file.read()
        
            self.data = re.search(rf'{data}\s*=\s*(.*)', content)
            if self.data:
                self.value_data = self.data.group(1)
            else:
                print("Không tìm thấy giá trị trong tệp.")

            if self.value_data == "True":
                self.value_data =True
            elif self.value_data == "False":
                self.value_data = False

            return self.value_data
        except Exception as ex:
            print("Không tìm thấy đề mục trong tệp.")
    
    def replace(self, data_re: str, data: str):
        with open(self.path, 'r') as file:
                content = file.read()
                w = content.split('\n')
        a = re.search(rf'{data}\s*=\s*(.*)', content)
        value_re = a.group(1)
        for i in range (len(w)):
            if w[i].__contains__(str(data)):
                w[i] = w[i].replace(value_re, data_re)
        try:
            with open(self.path, 'w') as file:
                for i in range (len(w)):
                    if i < len(w)-1: file.writelines(w[i] + '\n') 
                    else: file.writelines(w[i])
        except:
            print('not save')