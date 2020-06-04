from tkinter import messagebox, scrolledtext, Label, Button, Tk, INSERT,Toplevel, Text, font, Canvas, PhotoImage, NE,Image
from tkinter.ttk import Combobox
import classification
import os
 
# Поиск файлов в папке folder с расширением extencion 
# (должна быть папка Data и в ней файлы с расширением .csv)
def ScanFiles(folder, extension):
  files = os.listdir(folder)
  if files:
    files = list(filter(lambda x: x.endswith(extension), files))
    if folder == 'models':
        files.insert(0, 'Выберите модель')
    else:
        files.insert(0, 'Выберите файл')
  else: 
     if folder == 'models':
        files = list(['Выберите модель'])
     else:
        files = list(['Выберите файл'])
  return files 

# Списки для алгоритмов и файлов
#ALGORITHMS = ['Выберите алгоритм', 'Decision Tree', 'Super Vector Machine','Ridge Regression',
           #   'Lasso Regression','k-Nearest-Neighbors','Naive Bayes','Apriori','k-Means','t-SNE']
ALGORITHMS = ScanFiles('models','.sav')
FILES = ScanFiles('datasets', '.csv')
# Флаг запущена ли классификация в данный момент (чтобы 1000 окон нельзя было заспамить)
isClassify = False
# Создание главного окна
root = Tk()
# Выпадающие списки файлов и алгоритмов
algorithmsList = Combobox(root, values=ALGORITHMS)
algorithmsList.current(0)
filesList = Combobox(root, values=FILES)
filesList.current(0)
# Кнопка "Классификация"
classifyButton = Button(root,text="Классификация")


# Проверка на то, чтобы был выбран файл и алгоритм, иначе показывается ошибка 
# и не запустится классификация
def Check():
    if (algorithmsList.get() == ALGORITHMS[0]) and (filesList.get() == FILES[0]):
        messagebox.showerror('Ошибка', 'Выберите алгоритм и файл для классификации!')
        return False
    elif (algorithmsList.get() == ALGORITHMS[0]):
        messagebox.showerror('Ошибка!', 'Выберите алгоритм для классификации')
        return False
    elif (filesList.get() == FILES[0]):
        messagebox.showerror('Ошибка!', 'Выберите файл для классификации')
        return False
    elif isClassify:
        messagebox.showerror('Ошибка!', 'Классификация запущена!')
        return False
    else:
        return True


def Classify(key):
    
    if Check():
      window = Toplevel() 
      window.geometry('600x600')
      label = Label(window, text="Информация о наборе данных, модели и выбранных параметрах",font=font.Font(family='Helvetica', size=12, weight='bold'))  
      label.pack()
     
      result2 = Text(window,width=300, height=15)
      result2.pack()
      result1 = Text(window,width=300, height=3)
      result1.pack()
      
      label1 = Label(window, text="Классификация трафика",font=font.Font(family='Helvetica', size=12, weight='bold'))  
      label1.pack()
      
      result = scrolledtext.ScrolledText(window,width=300, height=35)
      result.pack()
      
      try:
          dataframe,predictions,best,model = classification.load_and_predict(algorithmsList.get(),filesList.get())
          dataframe1 = dataframe.iloc[:,:6]
          result2.insert(INSERT,f'                                             :{dataframe.head}\n')
          result1.insert(INSERT,f'\nВыбраная модель: {model}\nВыбранный набор параметров модели: {best}\n')
          for i in range(0,2):
              result.insert(INSERT,f'Индекс трафика: {i}\nОсновные параметры трафика:\n{dataframe1.iloc[i]}\n\nПредсказание:\nСтатус трафика: {predictions[i]}\n\n\n\n\n')
      except:
          result.insert(INSERT,f'wrong')
          
      
      
# Отрисовка элементов на экране
root.title('DDoS Classification with Machine Learning')
root.geometry('400x250')
classifyButton.bind('<ButtonPress>', Classify) # Бинд функции Classify по нажатию на кнопку
algorithmsList.pack()
filesList.pack()
classifyButton.pack()

root.mainloop()
