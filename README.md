# **AFTER** : **A F**ine-grained **T**raffic pr**E**diction f**R**amework


![alt text](img/traffic_ai.png "Traffic Research Team") ![alt text](img/DIETI.png "DIETI")

## *run_multiple_experiments_1step.py*, the main script to perform an experimental campaing
### **Input Parameters**
> **--confid (-c)** : configuration file to run an experiment (it will be edited automatically by the framework) 
>
> **--proto (-c)** : used to train a model at *PROTO* level on the specified transport-level protocol (choices={*TCP, UDP*})
>
> **--finter (-f)** : enables filtering based on app (a), activity (t) and on both (m) (choiches={*a, t, m*})
>
> **--output (-o)** : used to set the folder path to store the final results
>
> **--finter (-f)** : used to filter traffic based on app, activity, and both of them. If *a* is passed, it will train a model on the specified APP traffic only (APP-level model). If *t* is passed, it will train a model on the traffic of the specified ACTIVITY only. If *m* is passed, it will train a model only on traffic of the specified APP-ACTIVITY combination. *This parameter could be used in combination with the app parameter (the next one)*
>
> **--app (-a)** : used to specify the label that should be used to filter traffic, according to the filtering-level. It has to be used only when *finter* is used
>
> **--window (-w)** : used to set the window (memory) size (i.e., the number of packets used by the model to provide the predictions)
>
> **--gpu-id (-g)** : used to pass the ID of the GPU which must be used to run the experiments
>
> **--sensitivity (-s)** : enables the sensitivity analysis on the window size
>
> **--validation (-v)** : allows the use of a validation-set to perform training (i.e., early-stopping based on validation-loss, otherwise training-loss is used) 
>
> **--model (-m)** : used to specify the Deep Learning architecture. choice={*CNN, LSTM, GRU, SERIES_NET*}
>
> **--sampling (-S)** : enables the subsempling of the entire dataset (if used it uses the 10% of the dataset)


## Demo to launch an experiment

*python3 run_multiple_experiments_1step.py -c ./config_base.ini -e ./Experiments/ -o ./Experiments/FinalResults/ -v -m CNN -w 10*

## INFO
### *Tested on python3.8*
### The list of dependencies is provided in the *requirements.txt* file inside the root folder


## **Acknowledgements**
![alt text](img/vietsch-logo-e1490812981345.jpg "Vietsch Foundation")

This work is supported by the *“ADDITIONAL” Project* funded by **Vietsch Foundation**