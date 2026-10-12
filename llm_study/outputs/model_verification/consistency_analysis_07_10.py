"""Reproduce primary results with one metric definition and matched training data."""
from pathlib import Path
import sys, json, platform, importlib.metadata, warnings
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime_packages'))
sys.path.insert(1,str(HERE))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import balanced_accuracy_score,recall_score,f1_score
import critical_analysis as a
from models import evaluation
from models.model_registry import MODEL_REGISTRY
from models.preprocessing import TabularOneHotScaler

OUT=HERE/'consistency_07_10'
OUT.mkdir(exist_ok=True)
a.OUT=OUT
warnings.filterwarnings('ignore',message='X does not have valid feature names')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def report(s):
    print(s,flush=True)

def prepare(train,other,features):
    num=[f for f in a.META['feature_types']['numeric_features'] if f in features]
    cat=[f for f in a.META['feature_types']['categorical_features'] if f in features]
    p=TabularOneHotScaler(numeric_features=num,categorical_features=cat)
    x=p.fit_transform(train[features])
    return p,x,p.transform(other[features])

def metrics(y,pred,proba):
    values=a.metric_row(y,pred,proba)
    assert abs(values['balanced_accuracy']-recall_score(y,pred,average='macro'))<1e-12
    assert abs(evaluation.classification_metrics(y,pred)['balanced_accuracy']-values['balanced_accuracy'])<1e-12
    return values

def all_model_refits():
    train=a.DF[a.DF.Split=='train']
    val=a.DF[a.DF.Split=='validation']
    test=a.DF[a.DF.Split=='test']
    recorded=pd.read_csv(a.ROOT/'results/4.Hyperparameter Tuning/csv/tuned_validation_metrics_by_stage_model.csv')
    rows=[]
    for _,r in recorded.iterrows():
        features=a.META['stage_feature_sets'][r.stage]
        p,xtr,xval=prepare(train,val,features)
        xtest=p.transform(test[features])
        params=json.loads(r.best_params)
        overrides={}
        if r.model_key=='catboost': overrides={'allow_writing_files':False,'thread_count':4}
        if r.model_key in ('lightgbm','random_forest','xgboost'):overrides={'n_jobs':4}
        fitted=MODEL_REGISTRY[r.model_key].build(**params,**overrides)
        fitted.fit(xtr.to_numpy(),train.y.to_numpy())
        result={'stage':r.stage,'model':r.model,'model_key':r.model_key,'params':r.best_params,'n_train':len(train)}
        for split,x,y in [('validation',xval,val.y),('test',xtest,test.y)]:
            pred=fitted.predict(x.to_numpy()).reshape(-1)
            prob=fitted.predict_proba(x.to_numpy())
            result.update({split+'_'+k:v for k,v in metrics(y,pred,prob).items()})
        rows.append(result)
        report(f'Refit {r.stage} {r.model}: test F1={result["test_macro_f1"]:.6f}')
    table=pd.DataFrame(rows)
    table.to_csv(OUT/'all_model_refits.csv',index=False)
    selected=pd.read_csv(OUT/'selected_test_bootstrap_ci.csv')
    for _,r in selected.iterrows():
        main=table[(table.stage==r.stage)&(table.model_key=='lightgbm')].iloc[0]
        for metric in ['macro_f1','balanced_accuracy','log_loss']:
            assert abs(main['test_'+metric]-r[metric])<1e-10,(r.stage,metric)

def plot_stage_results():
    table=pd.read_csv(OUT/'all_model_refits.csv')
    styles=[('#2F6DAE','o','-'),('#C46A2D','s','--'),('#65783A','^','-.'),('#B58A22','D',':'),('#A7577A','v','--'),('#424952','P','-.')]
    keys=['random_forest','xgboost','lightgbm','catboost','rbf_svm','mlp']
    stages=list(a.SELECTED_PARAMS)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True)
    for ax,split,title in zip(axes,['validation','test'],['(a) Validation set (n = 885)','(b) Held-out test set (n = 442)']):
        for key,(color,marker,line) in zip(keys,styles):
            sub=table[table.model_key==key].set_index('stage').loc[stages]
            ax.plot(range(3),sub[split+'_macro_f1'],color=color,marker=marker,linestyle=line,label=sub.iloc[0]['model'],lw=2,ms=6)
        ax.set_xticks(range(3),['A\nEnrolment','B\nSemester 1','C\nSemester 2'])
        ax.set_title(title,fontsize=13)
        ax.grid(axis='y',color='#D9DDE2',alpha=.8)
        ax.set_ylim(.45,.78)
        ax.set_xlabel('Information stage')
    axes[0].set_ylabel('Macro-F1')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=11)
    fig.subplots_adjust(top=.76,bottom=.17,left=.08,right=.98,wspace=.12)
    fig.savefig(OUT/'figure3.png',dpi=240)
    plt.close(fig)

def plot_ablation():
    df=pd.read_csv(OUT/'fixed_lightgbm_ablation_test.csv')
    fig,ax=plt.subplots(figsize=(10,5.8))
    labels=['Academic only (14)','+ Demographic and financial (24)','+ Parental (18)','+ Macroeconomic (17)','+ Course and application (17)','All predictors (36)']
    x=df.macro_f1.to_numpy()
    err=np.vstack([x-df.macro_f1_ci_low,df.macro_f1_ci_high-x])
    ax.errorbar(x,np.arange(6),xerr=err,fmt='o',color='#2F6DAE',ecolor='#53606D',capsize=5,ms=7,lw=1.5)
    ax.set_yticks(np.arange(6),labels)
    ax.invert_yaxis()
    for i,value in enumerate(x):ax.text(value,i-.19,f'{value:.3f}',ha='center',fontsize=11)
    ax.set_xlim(.5,.8)
    ax.set_xlabel('Test Macro-F1 (95% bootstrap interval)')
    ax.set_title('Fixed LightGBM feature-group comparison',fontsize=14)
    ax.grid(axis='x',color='#D9DDE2')
    fig.tight_layout()
    fig.savefig(OUT/'figure4.png',dpi=240)
    plt.close(fig)

def short_name(s):
    return (s.replace('Curricular units 1st sem','Semester 1').replace('Curricular units 2nd sem','Semester 2')
            .replace('Tuition fees up to date','Tuition up to date').replace('Age at enrollment','Age at enrolment')
            .replace('Previous qualification (grade)','Prior qualification grade').replace('Daytime/evening attendance','Day/evening attendance'))

def shap_analysis():
    train=a.DF[a.DF.Split=='train'];test=a.DF[a.DF.Split=='test']
    specifications=[(stage,features,a.SELECTED_PARAMS[stage]) for stage,features in a.META['stage_feature_sets'].items()]
    specifications.append(('academic_financial',a.META['ablation_feature_sets']['academic_plus_financial_demographic'],a.SELECTED_PARAMS['C_second_semester']))
    rng=np.random.default_rng(123)
    all_rankings=[];panels=[]
    for name,features,params in specifications:
        prep,xtr,xt=prepare(train,test,features)
        fitted=a.model(params)
        fitted.fit(xtr.to_numpy(),train.y.to_numpy())
        explainer=shap.TreeExplainer(fitted,feature_perturbation='tree_path_dependent',model_output='raw')
        values=explainer.shap_values(xt.to_numpy())
        if isinstance(values,list):values=np.stack(values,axis=-1)
        dropout=np.asarray(values)[:,:,a.CLASS_TO_ID['Dropout']]
        grouped=np.zeros((len(test),len(features)))
        for j,feature in enumerate(features):
            columns=[i for i,c in enumerate(xt.columns) if c==feature or c.startswith(feature+'_')]
            assert columns,feature
            grouped[:,j]=dropout[:,columns].sum(axis=1)
        assert np.allclose(grouped.sum(axis=1),dropout.sum(axis=1))
        mean_abs=np.abs(grouped).mean(axis=0)
        rank=pd.DataFrame({'specification':name,'feature':features,'mean_abs_shap':mean_abs}).sort_values('mean_abs_shap',ascending=False)
        rank.to_csv(OUT/f'shap_{name}.csv',index=False)
        all_rankings.append(rank)
        panels.append((name,features,grouped,rank))
        if name=='C_second_semester':
            main=pd.read_csv(OUT/'selected_test_bootstrap_ci.csv').set_index('stage').loc[name]
            assert abs(f1_score(test.y,fitted.predict(xt.to_numpy()),average='macro')-main.macro_f1)<1e-12
            top=rank.head(10).iloc[::-1]
            fig,ax=plt.subplots(figsize=(9.5,6.2))
            ax.barh([short_name(v) for v in top.feature],top.mean_abs_shap,color='#2F6DAE')
            ax.set_xlabel('Mean absolute SHAP contribution to raw dropout score')
            ax.set_title('Stage C LightGBM dropout-class explanations',fontsize=14)
            ax.grid(axis='x',color='#D9DDE2');ax.set_axisbelow(True)
            fig.tight_layout();fig.savefig(OUT/'figure5.png',dpi=240);plt.close(fig)
    pd.concat(all_rankings).to_csv(OUT/'shap_all_rankings.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(13,11.2))
    titles=['(a) Enrolment information (Stage A)','(b) First-semester information (Stage B)','(c) Second-semester information (Stage C)','(d) Academic and financial-demographic']
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('value',['#2F6DAE','#DA8A36'])
    maxabs=max(np.abs(v[2][:,[v[1].index(f) for f in v[3].head(10).feature]]).max() for v in panels)
    for ax,(name,features,grouped,rank),title in zip(axes.flat,panels,titles):
        top=list(rank.head(10).feature)
        for row,feature in enumerate(top):
            vals=grouped[:,features.index(feature)]
            y=np.full(len(vals),9-row)+rng.normal(0,.075,len(vals))
            if feature in a.META['feature_types']['numeric_features']:
                c=test[feature].rank(pct=True).to_numpy()
                ax.scatter(vals,y,c=c,cmap=cmap,vmin=0,vmax=1,s=7,alpha=.65,rasterized=True,linewidths=0)
            else:ax.scatter(vals,y,color='#7B8188',s=7,alpha=.55,rasterized=True,linewidths=0)
        ax.axvline(0,color='#5D6570',lw=.8)
        ax.set_yticks(range(10),[short_name(v) for v in top[::-1]],fontsize=11.5)
        ax.set_title(title,fontsize=13,pad=10)
        ax.set_xlabel('SHAP contribution to raw dropout score',fontsize=11)
        ax.set_xlim(-maxabs*1.05,maxabs*1.05)
        ax.grid(axis='x',color='#E1E4E8',lw=.6)
    fig.subplots_adjust(left=.24,right=.99,bottom=.11,top=.95,wspace=.85,hspace=.3)
    fig.text(.5,.025,'Numerical values: blue = lower, orange = higher; categorical variables = grey.\nEach point represents one of the 442 held-out students.',ha='center',fontsize=11)
    fig.savefig(OUT/'figure6.png',dpi=240);plt.close(fig)

if __name__=='__main__':
    report('Primary selected-model estimates and paired intervals')
    a.selected_test_analysis()
    report('Repeated validation and baselines with matched training observations')
    a.repeated_cv_and_baselines()
    report('Fixed-model feature-group analysis')
    a.fixed_model_ablation()
    report('Refitting all recorded stage-model configurations')
    all_model_refits()
    report('Creating stage and ablation figures')
    plot_stage_results();plot_ablation()
    report('Computing consistent SHAP explanations')
    shap_analysis()
    environment={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','lightgbm','shap','xgboost','catboost','matplotlib']}
    environment.update(python=platform.python_version(),n_train=3097,n_validation=885,n_test=442,seed=123,metric='balanced_accuracy = macro recall',selection='Previously recorded validation-selected settings; no test-based reselection')
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    report('Analysis and consistency assertions complete')
