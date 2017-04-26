import cPickle as pickle
import pprint
import csv
import numpy as np
from sklearn.metrics import roc_auc_score

def parse_pickle(in_file, r_type, threshold=0):
    reader = pickle.load(open(in_file, 'rb'))

    p_score_lst = np.array([])
    y_true_lst = np.array([])

    for key in reader.keys():
        p_id = str(key[0])+"_"+str(key[1])+"_"+str(key[2])
        #print("\nPatient ID = " + p_id)
        img_dict = (reader[key]['images'])
        label = reader[key]['label']
        #print("Patient lb = " + label)
        y_true_lst = np.append(y_true_lst, int(label))
        if (r_type == 'mass'):
            mass_lst = img_dict['mass']
            #print(mass_lst)

            summed = 0
            for mass in mass_lst:
                mass_path = '../../dataset'+mass
                # print(mass_path)
                with open(r_type+"_result.csv", 'rb') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    for row in csvreader:
                        if(mass_path == row[0]):
                            summed += float(row[1])

                    csvfile.close()

            avg_score = summed/2
            p_score_lst = np.append(p_score_lst, avg_score)
            # print(str(p_id) + " score = " + str(avg_score) + "\n")

        elif (r_type == 'cal'):
            calc_lst = img_dict['calcification']
            if (threshold in [0.6, 0.7, 0.8, 0.9]):
                cal = calc_lst[str(threshold)]
                cal_path = '../../dataset'+cal
                #print(cal_path)
                score = 0
                with open(r_type+"_result.csv", 'rb') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    for row in csvreader:
                        if(cal_path == row[0]):
                            score = float(row[1])

                p_score_lst = np.append(p_score_lst, score)

            else:
                print("Invalid threshold! Please check the threshold value.")

    return p_score_lst, y_true_lst

if __name__ == "__main__":
    # Mass
    #p_sc_l, y_t_l = parse_pickle('crosswalk.p', 'mass', 0.6)
    #print(len(p_sc_l))
    #print(len(y_t_l))

    #AUC = roc_auc_score(y_t_l, p_sc_l)
    #print(AUC)

    # Calc
    for i in [0.6, 0.7, 0.8, 0.9]:
        p_sc_l, y_t_l = parse_pickle('crosswalk.p', 'cal', i)
        AUC = roc_auc_score(y_t_l, p_sc_l)
        print(AUC)
