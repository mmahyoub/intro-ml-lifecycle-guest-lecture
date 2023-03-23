from tkinter import *
import xgboost
import pandas as pd
import joblib 

data_schema = pd.read_csv('data_schema.csv')


def get_batch():
    global dataset

    f_loc = file_loc.get()

    dataset = pd.read_csv(f_loc) 

    process_batch()



def process_batch():
    global processed_data

    processed_data = {}

    #age
    processed_data['age'] = [a for a in dataset['age']]
    processed_data['sex'] = [ 1 if s == 'Male' else 0 for s in dataset['sex']]
    processed_data['hours-per-week'] = [h for h in dataset['hours-per-week']]
    processed_data['Is Married?'] = [1 if m in ('Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse') else 0 for m in dataset['Is Married?']]

    education_cols = [col for col in data_schema.columns if 'education' in col]

    for col in education_cols:
        for v in dataset['education']:
            if v in col:
                processed_data[col] = 1
            else:
                processed_data[col] = 0

    
    occupation_cols = [col for col in data_schema.columns if 'occupation' in col]

    for col in occupation_cols:
        for v in dataset['occupation']:
            if v in col:
                processed_data[col] = 1
            else:
                processed_data[col] = 0

    workclass_cols = [col for col in data_schema.columns if 'workclass' in col]

    for col in workclass_cols:
        for v in dataset['workclass']:
            if v in col:
                processed_data[col] = 1
            else:
                processed_data[col] = 0


    processed_data = pd.DataFrame(processed_data)

   


def run_model():
    xgb_loaded = joblib.load('./xgb_model.json')

    ypredict = xgb_loaded.predict(processed_data)

    output_data = {'ID':[i for i in range(len(ypredict))], 'Is > 50K?': [i for i in ypredict]}

    output_data = pd.DataFrame(output_data)

    output_data.to_csv('output_data.csv', index = False)

    print(output_data)

  
root = Tk()

root.title("Income Estimator by Mohammed Mahyoub")

root.geometry('500x200')

lbl1 = Label(root, text = "Insert data file location: ")
lbl1.grid()

file_loc = Entry(root, width = 40)
file_loc.grid(column=1, row=0)

data_btn = Button(root, text = 'Upload data', fg = 'red', command=get_batch)

data_btn.grid(column=2, row = 0)

model_btn = Button(root, text = 'Run model', fg = 'red', command= run_model)
model_btn.grid(column=1, row = 2)


root.mainloop()