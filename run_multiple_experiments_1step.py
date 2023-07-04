import configparser
import sys,os, getopt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import Traffic_prediction_per_fold_1step as runexp2

def run_concurrent_exp(cfg, config_file,dataset_path, app, n_features, model, granularity,
                       fold=10, output_path='.', n_samples=0,win_size=10, 
                       gpu_id=1, mode='EXP',
                       byproto=False, proto=None):
    

    #for current_fold in range(fold):
    local_cfg_file_name = os.path.join(os.path.dirname(config_file),'config_%s_W%s_%s.ini'%(model,win_size,app))
    #shutil.copy(config_file, local_cfg_file_name)
    cfg.set('BASE', 'app', app)
    cfg.set('BASE', 'n_features', str(n_features))
    cfg.set('BASE', 'win_size', str(win_size))
    cfg.set('I/O', 'app_from_dataset', dataset_path)
    cfg.set('I/O', 'output_path', output_path)
    cfg.set('MODEL', 'model', model)
    cfg.set('I/O', 'byproto',str(byproto))
    cfg.set('BASE', 'proto',str(proto))
    cfg.set('I/O', 'granularity', granularity)
    cfg.set('BASE', 'fold', str(fold))
    cfg.set('BASE', 'current_fold', str(0))
    cfg.set('BASE', 'n_samples', str(n_samples))

    out_feat=['L4_payload_bytes','iat_micros','packet_dir']

    
    cfg.set('COMPILE', 'pl_weight', str(0.45)) 
    cfg.set('COMPILE', 'iat_weight', str(0.1)) 
    cfg.set('COMPILE', 'pd_weight', str(0.45)) 
            
    cfg.set('BASE', 'out_fine_features', ','.join(out_feat))   
    print('INFO: Predicting %s'%out_feat)
    with open(local_cfg_file_name, 'w') as cfgfile:
        cfg.write(cfgfile)
        
    print('Config File : %s'%local_cfg_file_name)
    if mode=='EXP':
        runexp2.main(cfg,gpu_id)

#if __name__ == "__main__":
def main(argv):
    config_path='config_template.ini'
    finalres=None
    models=[]
    apps=[]
    sensitivity=False
    win=10
    gpu_id=None
    validation=False
    by_fold=False
    #appsx=['Zoom','Teams','Skype','Slack','Webex','Meets','Messenger','GotoMeeting','Discrod']
    appsx=['Zoom','Teams','Skype','Webex']
    #appsx=['Teams','Skype','Webex']
    activities=['audiocall','chat','videocall']
    mode='EXP'
    protosx=['UDP','TCP']
    protos=[]
    byproto=False
    filtering='X'
    sampling=False
    print(argv)
    try:
        opts, args = getopt.getopt(argv, "hc:m:o:e:f:sw:g:a:vp:S", "[source_dir_path=]")
    except getopt.GetoptError:
        print('<script_name.py> -o <OUTDIR> -e <EXP OUT> -m <MODEL> -c <CONFIG_FILE> -f <FILTER (a,t,m) -s <Sensitivity> -w < window size (int) -g <GPU-ID (int)> -a <app name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('HELP: <script_name.py> -o <OUTDIR> -m <MODEL> -c <CONFIG_FILE> -f <FILTER (a,t,m) -s <Sensitivity> -w <window size (int)> -g<GPU-ID (int)> -a <app name>')
            sys.exit()
        if opt in ('-c','--config'):
            config_path=arg
        if opt in ('-p','--proto'):
            byproto=True
            if arg in protosx:
                protos.append(arg)
        if opt in ('-f','--finter'):
            if arg in ['a']:
                filtering='app'
            elif arg in ['t']:
                filtering='activity'
            elif arg in ['m']:
                filtering='appactivity'   
            else:
                print('Wrong filter') 
                
        if opt in ('-o','--output'): 
            finalres=arg
        if opt in ('-e','--exppath'): 
            exppath=arg
        if opt in ('-w','--window'): 
            win=int(arg)
        if opt in ('-g','--gpu-id'): 
            gpu_id=arg
        if opt in ('-s','--sensitivity'): 
            sensitivity=True
        if opt in ('-v','--validation'): 
            validation=True
        if opt in ('-a','--app'):
            if arg in appsx:
                apps.append(arg)
        if opt in ('-S','--sampling'):
            sampling=True
        if opt in ('-m','--model'):
            if arg in ['CNN','LSTM','GRU','DSANET','GLOBAL_CNN_RNN','SERIES_NET','STN']:
                models.append(arg)
            else:
                print('Wrong model')
                exit()
                
    config = configparser.ConfigParser()
    config.read(config_path)
    #apps = ['Discord','GotoMeeting','Meets','Messenger','Skype','Slack','Teams','Webex','Zoom']
    if len(apps)==0:
        print('Experiment on all apps')
        apps=appsx
    else:
        print('Experiment on %s apps'%apps)
        
    if byproto:
        if len(protos)==0:
            print('Experiment on all apps')
            protos=protosx
        else:
            print('Experiment on %s protos'%protos)
    root_dir_datasets = config['I/O']['root_datasets']
    n_features = config['BASE'].getint('n_features')
    
    granularity = 'packets'
    if finalres is None:
        finalres = config['I/O']['final_res_root']
    total_fold = config['BASE'].getint('fold')
    
    if sensitivity:
        win_sizes=[2,5,10,20,30]
    else:
        win_sizes = [win]

            
    
    Path(exppath).mkdir(parents=True, exist_ok=True)         
    config.set('I/O', 'output_path',exppath)  
    
    if finalres is None:
        finalres=exppath
    else:
        Path(finalres).mkdir(parents=True, exist_ok=True)         
    config.set('I/O', 'final_res_root',finalres)       
            
    config.set('OPTIONS', 'validation_set',str(validation))
    config.set('OPTIONS', 'validation_split',str(0.2))
    config.set('OPTIONS', 'filtering',filtering+'proto' if byproto and filtering!='X' else 'proto' if filtering=='X' and byproto else 
               filtering)
    config.set('OPTIONS', 'sampling',str(sampling))
    n_samples = config['BASE'].getint('n_samples')

    if filtering=='app':
        for app in apps:
            for win_size in win_sizes:
                    for model in models:
                        dataset_path = root_dir_datasets
                        if byproto:
                            for proto in protos:
                                run_concurrent_exp( cfg=config, 
                                                    config_file=config_path,
                                                    dataset_path=dataset_path, 
                                                    app=app, 
                                                    n_features=n_features, 
                                                    model=model,
                                                    granularity=granularity,
                                                    fold=total_fold, 
                                                    output_path=exppath,
                                                    n_samples=n_samples, 
                                                    win_size=win_size,
                                                    gpu_id=gpu_id, 
                                                    mode=mode,  
                                                    byproto=byproto,
                                                    proto=proto)
                        else:
                            run_concurrent_exp( cfg=config, 
                                                config_file=config_path,
                                                dataset_path=dataset_path, 
                                                app=app, 
                                                n_features=n_features, 
                                                model=model,
                                                granularity=granularity,
                                                fold=total_fold, 
                                                output_path=exppath,
                                                n_samples=n_samples, 
                                                win_size=win_size,
                                                gpu_id=gpu_id, 
                                                mode=mode,  
                                                byproto=byproto)

    elif filtering=='activity':
        if classification_type is not None:
            classification_type='app'
            print('Model per ACTIVITY...Forcing to APP classification')
        for act in activities:
            for win_size in win_sizes:
                    for model in models:
                        dataset_path = root_dir_datasets
                        if byproto:
                            for proto in protos:
                                run_concurrent_exp( cfg=config, 
                                                    config_file=config_path,
                                                    dataset_path=dataset_path, 
                                                    app=act, 
                                                    n_features=n_features, 
                                                    model=model,
                                                    granularity=granularity,
                                                    fold=total_fold, 
                                                    output_path=exppath,
                                                    n_samples=n_samples, 
                                                    win_size=win_size,
                                                    gpu_id=gpu_id, 
                                                    mode=mode, 
                                                    byproto=byproto,
                                                    proto=proto)
                        else:
                            run_concurrent_exp(
                                                    cfg=config, 
                                                    config_file=config_path,
                                                    dataset_path=dataset_path, 
                                                    app=act, 
                                                    n_features=n_features, 
                                                    model=model,
                                                    granularity=granularity,
                                                    fold=total_fold, 
                                                    output_path=exppath,
                                                    n_samples=n_samples, 
                                                    win_size=win_size,
                                                    gpu_id=gpu_id, 
                                                    mode=mode, 
                                                    byproto=byproto)
    
    
    else:
        
        with open(config_path, 'w') as cfgfile:
            config.write(cfgfile)
            
        if not by_fold:
            for win_size in win_sizes:
                    for model in models:
                        output_path = config['I/O']['output_path']
                        dataset_path = root_dir_datasets

                        if byproto:
                            for proto in protos:
                                run_concurrent_exp(
                                                    cfg=config, 
                                                    config_file=config_path,
                                                    dataset_path=dataset_path, 
                                                    app='ALL', 
                                                    n_features=n_features, 
                                                    model=model,
                                                    granularity=granularity,
                                                    fold=total_fold, 
                                                    output_path=exppath,
                                                    n_samples=n_samples, 
                                                    win_size=win_size,
                                                    gpu_id=gpu_id, 
                                                    mode=mode, 
                                                    byproto=byproto,
                                                    proto=proto,
                                                    )
                        else:
                            run_concurrent_exp(
                                                cfg=config, 
                                                config_file=config_path,
                                                dataset_path=dataset_path, 
                                                app='ALL', 
                                                n_features=n_features, 
                                                model=model,
                                                granularity=granularity,
                                                fold=total_fold, 
                                                output_path=exppath,
                                                n_samples=n_samples, 
                                                win_size=win_size,
                                                gpu_id=gpu_id, 
                                                mode=mode,
                                                byproto=byproto)

        else:
            print('ToDo')       
if __name__ == '__main__':
    main(sys.argv[1:])