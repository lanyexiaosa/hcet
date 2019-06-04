
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from datetime import timedelta


# import data
path_demo=""# path for demographics information
input_demo=pd.read_csv(path_demo,header=None)
data_demo=pd.DataFrame(input_demo).values

path_topic="" #path for topic features
input_topic=pd.read_csv(path_topic,header=0)
data_topic=pd.DataFrame(input_topic).values

#import table of ICD-9 code
path_icd="" #path for ICD-9 code data
input_icd=pd.read_csv(path_icd,header=0)
data_icd=pd.DataFrame(input_icd).values

#import table of cpt code
path_cpt="" #path for cpt code data
input_cpt=pd.read_csv(path_cpt,header=0)
data_cpt=pd.DataFrame(input_cpt).values

#import table of medication
path_med="" #path for medication
input_med=pd.read_csv(path_med,header=0)
data_med=pd.DataFrame(input_med).values

#import table of significant feature for ICD-9 code
path_sficd="" #path for significant features of ICD-9 code
input_sficd=pd.read_csv(path_sficd,header=None)
sf_icd=pd.DataFrame(input_sficd).values # significant feature of ICD-9 code
sf_icd=list(set(sf_icd[:,0]))

#import table of significant feature for cpt code
path_sfcpt="" #path for significant features of cpt code
input_sfcpt=pd.read_csv(path_sfcpt,header=None)
sf_cpt=pd.DataFrame(input_sfcpt).values # significant feature of cpt code
sf_cpt=list(set(sf_cpt[:,0]))

#import table of significant feature for medication
path_sfmed="" #path for significant features of medication
input_sfmed=pd.read_csv(path_sfmed,header=None)
sf_med=pd.DataFrame(input_sfmed).values # significant feature of medication
sf_med=list(set(sf_med[:,0]))


#get patient number from file of baseline model
path_pn="" #path of patient number from baseline model
pndata=pd.read_csv(path_pn,header=None)
pnind=pd.DataFrame(pndata).values
pn_all=pnind[:,0].astype(int)
pn_label=pnind[:,-1].astype(int)

ind1=np.where(pn_label==1)
pn_pos=pn_all[ind1]
ind2=np.where(pn_label==0)
pn_neg=pn_all[ind2]

# get diagnosis time for patient with depression
path_depre_time="" #path for a file containing diagnosis time of depression
time_data=pd.read_csv(path_depre_time,header=0)
depre_time=pd.DataFrame(time_data).values


# get sequence code for each patients
def get_data(back_day,block,window):
        
    pairs=[]
    patient_data=[]
    labels=[]
    pids=[]
    n_visits=0
    max_icd_per_visit=0
    max_med_per_visit=0
    max_cpt_per_visit=0
    max_topic_per_visit=0
    icd_count=0
    cpt_count=0
    med_count=0
    topic_count=0

    for n in pn_pos:
        #diagnosis time of depression
        pn_depre_time=datetime.strptime(str(np.squeeze(depre_time[np.where(depre_time[:,0]==n),1])),'%Y-%m-%d %H:%M:%S') 

        #table of patient data with ICD-9 code
        icd_ind=np.where(data_icd[:,0]==n)[0]
        pn_icd=data_icd[icd_ind,:]
        pn_icd_times=[datetime.strptime(pntime,'%Y-%m-%d %H:%M:%S') for pntime in pn_icd[:,1]]

        #table of patient data with cpt code
        cpt_ind=np.where(data_cpt[:,0]==n)[0]
        pn_cpt=data_cpt[cpt_ind,:]
        pn_cpt_times=[datetime.strptime(pntime.replace("'",""),'%Y-%m-%d %H:%M:%S') for pntime in pn_cpt[:,1]]

        #table of patient data with medication code
        med_ind=np.where(data_med[:,0]==n)[0]
        pn_med=data_med[med_ind,:]
        pn_med_times=[datetime.strptime(pntime.replace("'",""),'%Y-%m-%d %H:%M:%S') for pntime in pn_med[:,1]]
        
        #table of patient data with medication code
        topic_ind=np.where(data_topic[:,0]==n)[0]
        pn_topic=data_topic[topic_ind,:]
        pn_topic_times=[datetime.strptime(pntime,'%Y-%m-%d %H:%M:%S') for pntime in pn_topic[:,1]]

        #find visit times in the 6 month window
        window_icd_times=[atime for atime in pn_icd_times if atime<pn_depre_time+timedelta(-block-back_day) and 
                   atime >= pn_depre_time+timedelta(-block-back_day-window)]
        window_cpt_times=[atime for atime in pn_cpt_times if atime<pn_depre_time+timedelta(-block-back_day) and 
                   atime >= pn_depre_time+timedelta(-block-back_day-window)]
        window_med_times=[atime for atime in pn_med_times if atime<pn_depre_time+timedelta(-block-back_day) and 
                   atime >= pn_depre_time+timedelta(-block-back_day-window)]
        window_topic_times=[atime for atime in pn_topic_times if atime<pn_depre_time+timedelta(-block-back_day) and 
                   atime >= pn_depre_time+timedelta(-block-back_day-window)]

        #all unique time from EHR

        pn_alltime=set([])
        for atime in set(window_icd_times):
            pn_alltime.add(atime)
        for atime in set(window_cpt_times):
            pn_alltime.add(atime)
        for atime in set(window_med_times):
            pn_alltime.add(atime) 
        for atime in set(window_topic_times):
            pn_alltime.add(atime)    
        pn_alltime=list(pn_alltime) 
        pn_alltime.sort()

        patient_seq=[]

        for atime in pn_alltime:
            visit=[]
            visit_icd_count=0
            visit_cpt_count=0
            visit_med_count=0
            visit_topic_count=0
            #get ICD-9 code
            icd_code=pn_icd[np.where(pn_icd_times==np.datetime64(atime))[0],2]
            if len(icd_code)>0:
                icd_code=list(set(icd_code))
                for code in icd_code:
                    if code in sf_icd:
                        #visit.append(sf_icd.index(code))
                        visit.append(sf_icd.index(code)+85)
                        icd_count+=1
                        visit_icd_count+=1
                max_icd_per_visit=max(max_icd_per_visit,visit_icd_count)

            #get cpt code
            cpt_code=pn_cpt[np.where(pn_cpt_times==np.datetime64(atime))[0],2]
            if len(cpt_code)>0:
                cpt_code=list(set(cpt_code))
                for code in cpt_code:
                    if code in sf_cpt:
                        #visit.append(sf_cpt.index(code)+199)
                        visit.append(sf_cpt.index(code)+284)
                        cpt_count+=1
                        visit_cpt_count+=1
                max_cpt_per_visit=max(max_cpt_per_visit,visit_cpt_count)        

            #get medication code
            if len(pn_med_times)>0:
                med_code=pn_med[np.where(pn_med_times==np.datetime64(atime))[0],2]
                if len(med_code)>0:
                    med_code=list(set(med_code))
                    for code in med_code:
                        if code in sf_med:
                            #visit.append(sf_med.index(code)+2197)
                            visit.append(sf_med.index(code)+2282)
                            med_count+=1
                            visit_med_count+=1
                    max_med_per_visit=max(max_med_per_visit,visit_med_count)
            
            #get topic features
            if len(pn_topic_times)>0:
                #print(n)
                topic_code=np.squeeze(pn_topic[np.where(pn_topic_times==np.datetime64(atime))[0],2:])
                if len(topic_code)>0:
                 #   print(atime)
                  #  print(topic_code)
                    topic_code=list(set(topic_code))
                    for code in topic_code:
                        #visit.append(sf_med.index(code)+2197) #use this line if no demographic data and topic feature
                        #visit.append(sf_med.index(code)+2347) #use this line if no topic feature
                        if code>-1:
                            visit.append(code+2773)
                            topic_count+=1
                            visit_topic_count+=1
                    max_topic_per_visit=max(max_topic_per_visit,visit_topic_count)
                    
            #add visit code for patient
            if len(visit)>0:
                # add demo code to the end of the visit
                demo_ind=np.where(data_demo[:,0]==n)[0]
                pn_demo=data_demo[demo_ind,1:][0]
                visit.extend(pn_demo)
                
                #add visit code for patient
                patient_seq.append(visit)  
                n_visits+=1

        if len(patient_seq)>0:
            patient_data.append(patient_seq)
            labels.append(1)
            pids.append(n)
            pairs.append((n,len(patient_seq)))
            #print("Patient ID: %d , number of visit: %d" % (n,len(patient_seq)))
    
    #generate sequence for non-depressed patients
    for n in pn_neg:
    # n=12164
        #table of patient data with ICD-9 code
        icd_ind=np.where(data_icd[:,0]==n)[0]
        pn_icd=data_icd[icd_ind,:]
        pn_icd_times=[datetime.strptime(pntime,'%Y-%m-%d %H:%M:%S') for pntime in pn_icd[:,1]]
        last_time_icd=max(pn_icd_times)

        #table of patient data with cpt code
        cpt_ind=np.where(data_cpt[:,0]==n)[0]
        pn_cpt=data_cpt[cpt_ind,:]
        pn_cpt_times=[datetime.strptime(pntime.replace("'",""),'%Y-%m-%d %H:%M:%S') for pntime in pn_cpt[:,1]]
        if len(pn_cpt_times)>0:
            last_time_cpt=max(pn_cpt_times)
        else:
            last_time_cpt=last_time_icd      

        #table of patient data with medication code
        med_ind=np.where(data_med[:,0]==n)[0]
        pn_med=data_med[med_ind,:]
        pn_med_times=[datetime.strptime(pntime.replace("'",""),'%Y-%m-%d %H:%M:%S') for pntime in pn_med[:,1]]
        if len(pn_med_times)>0:
            last_time_med=max(pn_med_times)
        else:
            last_time_med=last_time_icd
        
        #table of patient data with topic feature
        topic_ind=np.where(data_topic[:,0]==n)[0]
        pn_topic=data_topic[topic_ind,:]
        pn_topic_times=[datetime.strptime(pntime,'%Y-%m-%d %H:%M:%S') for pntime in pn_topic[:,1]]
        if len(pn_topic_times)>0:
            last_time_topic=max(pn_topic_times)
        else:
            last_time_topic=last_time_icd

        end_time=max(last_time_icd,last_time_cpt,last_time_med,last_time_topic)
        
        #find visit times in the 6 month window
        window_icd_times=[atime for atime in pn_icd_times if atime<end_time+timedelta(-block-back_day) and 
                   atime >= end_time+timedelta(-block-back_day-window)]
        window_cpt_times=[atime for atime in pn_cpt_times if atime<end_time+timedelta(-block-back_day) and 
                   atime >= end_time+timedelta(-block-back_day-window)]
        window_med_times=[atime for atime in pn_med_times if atime<end_time+timedelta(-block-back_day) and 
                   atime >= end_time+timedelta(-block-back_day-window)]
        window_topic_times=[atime for atime in pn_topic_times if atime<end_time+timedelta(-block-back_day) and 
                   atime >= end_time+timedelta(-block-back_day-window)]
        #all unique time from EHR

        pn_alltime=set([])
        for atime in set(window_icd_times):
            pn_alltime.add(atime)
        for atime in set(window_cpt_times):
            pn_alltime.add(atime)
        for atime in set(window_med_times):
            pn_alltime.add(atime) 
        for atime in set(window_topic_times):
            pn_alltime.add(atime) 
        pn_alltime=list(pn_alltime) 
        pn_alltime.sort()

        patient_seq=[]

        for atime in pn_alltime:
            visit=[]
            visit_icd_count=0
            visit_cpt_count=0
            visit_med_count=0
            visit_topic_count=0
            #get ICD-9 code
            if len(pn_icd_times)>0:
                icd_code=pn_icd[np.where(pn_icd_times==np.datetime64(atime))[0],2]
                if len(icd_code)>0:
                    icd_code=list(set(icd_code))
                    for code in icd_code:
                        if code in sf_icd:
                            #visit.append(sf_icd.index(code))
                            visit.append(sf_icd.index(code)+85)
                            icd_count+=1
                            visit_icd_count+=1
                    max_icd_per_visit=max(max_icd_per_visit,visit_icd_count)        

            #get cpt code
            if len(pn_cpt_times)>0:
                cpt_code=pn_cpt[np.where(pn_cpt_times==np.datetime64(atime))[0],2]
                if len(cpt_code)>0:
                    cpt_code=list(set(cpt_code))
                    for code in cpt_code:
                        if code in sf_cpt:
                            #visit.append(sf_cpt.index(code)+199) 
                            visit.append(sf_cpt.index(code)+284)
                            cpt_count+=1
                            visit_cpt_count+=1
                    max_cpt_per_visit=max(max_cpt_per_visit,visit_cpt_count)    

            #get medication code
            if len(pn_med_times)>0:
                med_code=pn_med[np.where(pn_med_times==np.datetime64(atime))[0],2]
                if len(med_code)>0:
                    med_code=list(set(med_code))
                    for code in med_code:
                        if code in sf_med:
                            #visit.append(sf_med.index(code)+2197)
                            visit.append(sf_med.index(code)+2282)
                            med_count+=1
                            visit_med_count+=1
                    max_med_per_visit=max(max_med_per_visit,visit_med_count)
                    
            #get topic features
            if len(pn_topic_times)>0:
                topic_code=np.squeeze(pn_topic[np.where(pn_topic_times==np.datetime64(atime))[0],2:])
                if len(topic_code)>0:
                    topic_code=list(set(topic_code))
                    for code in topic_code:
                        #visit.append(sf_med.index(code)+2197) #use this line if no demographic data and topic feature
                        #visit.append(sf_med.index(code)+2347) #use this line if no topic feature
                        if code>-1:
                            visit.append(code+2773)
                            topic_count+=1
                            visit_topic_count+=1
                    max_topic_per_visit=max(max_topic_per_visit,visit_topic_count)
                    
                    
            #add visit code for patient
            if len(visit)>0:
                # add demo code to the end of the visit
                demo_ind=np.where(data_demo[:,0]==n)[0]
                pn_demo=data_demo[demo_ind,1:][0]
                visit.extend(pn_demo)
                
                patient_seq.append(visit)
                n_visits+=1

        if len(patient_seq)>0:
            patient_data.append(patient_seq)
            labels.append(0) #label 0 for non-depressed patients
            pids.append(n)
            pairs.append((n,len(patient_seq)))
            #print("Patient ID: %d , number of visit: %d" % (n,len(patient_seq)))
            
    print("length of patient sequence:",len(patient_data))
    print("number of labels:",len(labels))
    print("number of visits:",n_visits)
    print("max_icd_per_visit:",max_icd_per_visit)
    print("max_cpt_per_visit:",max_cpt_per_visit)
    print("max_med_per_visit:",max_med_per_visit)
    print("max_topic_per_visit:",max_topic_per_visit)
    print("icd_count:",icd_count)
    print("cpt_count:",cpt_count)
    print("med_count:",med_count)
    print("topic_count:",topic_count)
    print("length of pids:",len(pids))
    
    max_visit=0
    min_visit=200

    for i,seq in enumerate(patient_data):
        if len(seq)>max_visit:
            max_visit=len(seq)
        if len(seq)<min_visit:
            min_visit=len(seq)

    print("max number of visits:",max_visit)
    print("min number of visits:",min_visit)
    
    return patient_data,labels,pids,pairs


if __name__='__main__':

    days_data,days_label,days_pids,_=get_data(15,0,180)
    #save the data
    pickle.dump(days_pids, open('data_15days_all.pids', 'wb'),-1)
    pickle.dump(days_label, open('data_15days_all.labels', 'wb'), -1)
    pickle.dump(days_data,open('data_15days_all.seqs', 'wb'), -1)

    threemonth_data,threemonth_label,threemonth_pids,_=get_data(15,90,180)
    #save the data
    pickle.dump(threemonth_pids, open('data_threemonth_all.pids', 'wb'),-1)
    pickle.dump(threemonth_label, open('data_threemonth_all.labels', 'wb'), -1)
    pickle.dump(threemonth_data,open('data_threemonth_all.seqs', 'wb'), -1)


    sixmonth_data,sixmonth_label,sixmonth_pids,_=get_data(15,180,180)
    #save the data
    pickle.dump(sixmonth_pids, open('data_sixmonth_all.pids', 'wb'),-1)
    pickle.dump(sixmonth_label, open('data_sixmonth_all.labels', 'wb'), -1)
    pickle.dump(sixmonth_data,open('data_sixmonth_all.seqs', 'wb'), -1)

    oneyear_data,oneyear_label,oneyear_pids,_=get_data(15,365,180)
    #save the data
    pickle.dump(oneyear_pids, open('data_oneyear_all.pids', 'wb'),-1)
    pickle.dump(oneyear_label, open('data_oneyear_all.labels', 'wb'), -1)
    pickle.dump(oneyear_data,open('data_oneyear_all.seqs', 'wb'), -1)

