from tkinter import *
from tkinter.font import *
from evaluator import *
from similarity import *
from tkinter import filedialog
from xlrd import open_workbook
from xlwt import Workbook

class baseframe():
    def __init__(self, window):
        self.window = window
        self.frame = Frame(window)
        self.entrys = []
        self.initframe()       
        self.frame.pack()
    
    def get_entry(self, root, content):
        self.entrys = []
        ft = Font(size=18)
        column = 3
        for i, item in enumerate(content):
            label = Label(root, text=item, width=20, height=2, font=ft).grid(row=i, column=column)
            entry = Entry(root, width=40, font=ft)
            entry.grid(row=i, column=column+1)
            self.entrys.append(entry)
        return self.entrys
    
    def init_entry(self, entrys, init):
        for i, item in enumerate(init):
            self.entrys[i].insert(10, item)
    
    def cal_bleu(self):
        standard_output = self.entrys[0].get()
        predict_output = self.entrys[1].get()
        n = self.entrys[2].get()
        weights = self.entrys[3].get()
        self.entrys[4].delete(0, END)
        self.entrys[5].delete(0, END)
        self.entrys[6].delete(0, END)
    
        try:
            n = int(n)
            try:
                tmp_weights = str(weights).strip().split()
                weights = []
                for weight in tmp_weights:
                    weights.append(float(weight))
                self.entrys[4].insert(10, str(evaluator(standard_output, [predict_output], n, weights)))
            except:
                self.entrys[4].insert(10, "you should enter correct weights (list of float, eg. 0.25 0.25 0.25 0.25)")
        except:
            self.entrys[4].insert(10, "you should enter correct n (int, eg. 4)")
        self.entrys[5].insert(10, str(calculate_exactmatch(standard_output, predict_output)))
        self.entrys[6].insert(10, str(calculate_f1score(standard_output, predict_output)))
    
    def changeframe(self):
        self.frame.destroy()
        fileframe(self.window)
        
    def initframe(self):
        
        content = ["standard output:", "predict output:", "n(bleu only):", "weights(bleu only):", "bleu-result:", "exactmatch-result:", "f1score-result:"]
        init = ["please enter your standard output", "please enter your predict output", "4", "0.25 0.25 0.25 0.25"]
        
        self.entrys = self.get_entry(self.frame, content)
        self.init_entry(self.entrys, init)
        
        ft = Font(size=18)
        column=3
        button1 = Button(self.frame, text="start", font=ft, command=self.cal_bleu).grid(row=len(self.entrys)+1, column=column)
        button2 = Button(self.frame, text="quit", font=ft, command=self.window.destroy).grid(row=len(self.entrys)+1, column=column+1)
        
        button3 = Button(self.frame, text="upload file", font=ft, command=self.changeframe).grid(row=len(self.entrys)+1, column=column+2)

class fileframe():
    def __init__(self, window):
        self.window = window
        self.frame = Frame(window)
        self.entry = ""
        self.initframe()       
        self.frame.pack()
        
    def initframe(self):
        row = 5
        column = 3
        self.adjust_pos(row)
        ft = Font(size=18)

        self.entry = Entry(self.frame, width=40, font=ft)
        self.entry.grid(row=row+1, column=column)
        self.entry.insert(10, "select a filepath such as xx.xlsx")
        label = Label(self.frame, text="").grid(row=row+2, column=column)
        label = Label(self.frame, text="").grid(row=row+2, column=column+1)
        button1 = Button(self.frame, text="select file", font=ft, command=self.upload_file).grid(row=row+1, column=column+1)
        button2 = Button(self.frame, text="start", font=ft, command=self.cal_bleu).grid(row=row+3, column=column)
        
        button3 = Button(self.frame, text="back", font=ft, command=self.changeframe).grid(row=row+3, column=column+1)
    
    def cal_bleu(self):
        filepath = self.entry.get()
        try:
            workbook = open_workbook(filepath)
            sheet = workbook.sheets()[0]
            nrows = sheet.nrows
            
            writebook = Workbook()
            writesheet = writebook.add_sheet('result')
            writesheet.write(0, 0, "standard_output")
            writesheet.write(0, 1, "predict_output")
            writesheet.write(0, 2, "n")
            writesheet.write(0, 3, "weights")
            writesheet.write(0, 4, "bleu-result")
            writesheet.write(0, 5, "exactmatch-result")
            writesheet.write(0, 6, "f1score-result")
            
            for i in range(nrows):
                standard_output = sheet.cell(i, 0).value
                predict_output = sheet.cell(i, 1).value
                n = sheet.cell(i, 2).value
                weights = sheet.cell(i, 3).value
                writesheet.write(i+1, 0, standard_output)
                writesheet.write(i+1, 1, predict_output)
                writesheet.write(i+1, 2, n)
                writesheet.write(i+1, 3, weights)
                try:
                    n = int(n)
                    try:
                        
                        tmp_weights = str(weights).strip().split()
                        
                        weights = []
                        for weight in tmp_weights:
                            weights.append(float(weight))
                        writesheet.write(i+1, 4, str(evaluator(standard_output, [predict_output], n, weights)))
                    except:
                        writesheet.write(i+1, 4, "you should enter correct weights (list of float, eg. 0.25 0.25 0.25 0.25)")
                except:
                    writesheet.write(i+1, 4, "you should enter correct n (int, eg. 4)")
                writesheet.write(i+1, 5, str(calculate_exactmatch(standard_output, predict_output)))
                writesheet.write(i+1, 6, str(calculate_f1score(standard_output, predict_output)))
            writebook.save("result.xls")
            
        except:
            self.entry.delete(0, END)
            self.entry.insert(10, "you should select correct file")
        
            
    def adjust_pos(self, n):
        for i in range(n):
            label = Label(self.frame, text="").grid(row=i)
            
    def upload_file(self):
        self.entry.delete(0, END)
        selectFile = filedialog.askopenfilename()
        self.entry.insert(0, selectFile)                

    def changeframe(self):
        self.frame.destroy()
        baseframe(self.window)

if __name__ == '__main__':
    window = Tk()
    window.title("welcome to bleu-n evaluator")
    window.geometry("1000x450")
    window.resizable(0,0)
    baseframe(window)
    window.mainloop()