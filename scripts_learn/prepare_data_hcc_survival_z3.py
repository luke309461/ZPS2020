 


def save_data(x_train , y_train, x_test , y_test, classes_names, output_dir):
    

    # zapisujemy dane treningowe
    x_train_all_dict  =    {     'data': x_train ,
                                    'classes':y_train}
                
    train_data_outfile = open(output_dir + '/train_data.pkl', 'wb')
    pickle.dump(x_train_all_dict, train_data_outfile)



    # zapisujemy dane testowe 
    x_test_all_dict  =  {'data': x_test  ,
                        'classes':y_test}
     
    test_data_outfile = open(output_dir + '/test_data.pkl', 'wb')
    pickle.dump(x_test_all_dict, test_data_outfile)

    # zapisujemy nazwy klas
    cl_names_outfile = open(output_dir + '/class_names.pkl', 'wb')
    pickle.dump(classes_names, cl_names_outfile)

    print("Pickles saved in ", output_dir)
    
    


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")
    parser.add_argument('--input-dir', default="", required=True, help='data dir (default: %(default)s)')
    parser.add_argument('--output-dir', default="", required=True, help='output dir (default: %(default)s)')
    parser.add_argument('--fraction', default="0.2", required=False, help='size of test set (fration) (default: %(default)s)')
    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.fraction
        
input_dir, output_dir, test_size_fraction   = ParseArguments()

test_size_fraction = float(test_size_fraction)

# wczytaj plik csv
df = pd.read_csv(input_dir+"/hcc_survival_z3_data.txt", sep=",")

#JEST 50 kolumn, ostatnia to - 50. to klasyfikator

# Zamieniamy DataFrame (df) na macierz numpy

data_all = df.to_numpy()

# pierwsze 49 kolumn to dane:

data = data_all[:,:49] # = wszystkie wiersze, kolumny do 49.

# ostatnia kolumna to klasy:

data_classes = data_all[:,50] # = wszystkie wiersze, kolumna 50

# nazwy klas -- damy tutaj wszystkie unikalne numery, ktore wystepuja w data_classes
# (mozna np. reczni wpisac classes_names = ['divorced', 'not divorced']

classes_names = np.unique(data_classes)
 

# dzielimy zbior na treningowy i testowy

x_train, x_test, y_train, y_test = train_test_split(data, data_classes,
            test_size=test_size_fraction, random_state=42)
            

save_data(x_train , y_train, x_test , y_test, classes_names, output_dir)
    