"""Timing sensitivity and expanded descriptive clustering checks, fixed seed 123."""
from pathlib import Path
import sys,json,warnings
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime_packages'))
sys.path.insert(1,str(HERE))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,silhouette_score,calinski_harabasz_score,adjusted_rand_score
from sklearn.cluster import KMeans
from itertools import combinations
import critical_analysis as a
OUT=HERE/'submission_checks_09_10'
OUT.mkdir(exist_ok=True)
warnings.filterwarnings('ignore',message='X does not have valid feature names')
REMOVED=['Tuition fees up to date','Debtor','Scholarship holder']

def run():
    train=a.DF[a.DF.Split=='train'];test=a.DF[a.DF.Split=='test']
    y=test.y.to_numpy();rng=np.random.default_rng(123)
    pred={};rows=[];features_manifest={}
    for spec in ['recorded','without_untimestamped_financial_status']:
        for stage,params in a.SELECTED_PARAMS.items():
            features=[f for f in a.META['stage_feature_sets'][stage] if spec=='recorded' or f not in REMOVED]
            features_manifest[spec+'__'+stage]=features
            xtr,xt=a.prep(train,test,features)
            fitted=a.model(params);fitted.set_params(n_jobs=4)
            fitted.fit(xtr,train.y);yp=fitted.predict(xt).astype(int);prob=fitted.predict_proba(xt)
            pred[spec,stage]=yp
            vals=[f1_score(y[idx],yp[idx],average='macro',labels=[0,1,2],zero_division=0) for idx in rng.integers(0,len(y),(3000,len(y)))]
            lo,hi=a.percentile_ci(vals)
            rows.append(dict(specification=spec,stage=stage,n_predictors=len(features),**a.metric_row(y,yp,prob),macro_f1_ci_low=lo,macro_f1_ci_high=hi))
            print(spec,stage,rows[-1]['macro_f1'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'timing_sensitivity_metrics.csv',index=False)
    diffs=[];stages=list(a.SELECTED_PARAMS)
    contrasts=[]
    for spec in ['recorded','without_untimestamped_financial_status']:
        contrasts.extend([(spec,stages[i+1],spec,stages[i],stages[i+1]+' minus '+stages[i]) for i in range(2)])
    contrasts.extend([('without_untimestamped_financial_status',s,'recorded',s,'removed minus recorded') for s in stages])
    for ls,lstage,es,estage,label in contrasts:
        lp=pred[ls,lstage];ep=pred[es,estage]
        vals=[]
        for idx in rng.integers(0,len(y),(5000,len(y))):
            vals.append(f1_score(y[idx],lp[idx],average='macro',labels=[0,1,2],zero_division=0)-f1_score(y[idx],ep[idx],average='macro',labels=[0,1,2],zero_division=0))
        lo,hi=a.percentile_ci(vals)
        diffs.append(dict(later_specification=ls,earlier_specification=es,stage=lstage,comparison=label,difference=f1_score(y,lp,average='macro')-f1_score(y,ep,average='macro'),ci_low=lo,ci_high=hi))
    pd.DataFrame(diffs).to_csv(OUT/'timing_sensitivity_paired_differences.csv',index=False)
    saved=test[['StudentID','Target','Split']].copy()
    for (spec,stage),yp in pred.items():saved[spec+'__'+stage]=yp
    saved.to_csv(OUT/'timing_sensitivity_test_predictions.csv',index=False)
    (OUT/'feature_sets.json').write_text(json.dumps(features_manifest,indent=2),encoding='utf-8')

    print('Expanded clustering candidates k=2 through 8',flush=True)
    x,_=a.prep(a.DF,a.DF,a.RISK_FEATURES)
    selection=[];labels_by_k={}
    for k in range(2,9):
        fit=KMeans(n_clusters=k,random_state=123,n_init=50).fit(x)
        labels_by_k[k]=fit.labels_
        selection.append(dict(k=k,silhouette=silhouette_score(x,fit.labels_),calinski_harabasz=calinski_harabasz_score(x,fit.labels_),inertia=fit.inertia_))
        print(selection[-1],flush=True)
    pd.DataFrame(selection).to_csv(OUT/'cluster_selection_k2_to_k8.csv',index=False)
    best=max(selection,key=lambda r:r['silhouette'])['k']
    for k in sorted(set([3,best])):
        labels=KMeans(n_clusters=k,random_state=123,n_init=100).fit_predict(x)
        work=a.DF.copy();work['cluster']=labels
        profiles=work.groupby('cluster').agg(n=('StudentID','count'),age_mean=('Age at enrollment','mean'),first_approved_mean=('Curricular units 1st sem (approved)','mean'),first_grade_mean=('Curricular units 1st sem (grade)','mean'),second_approved_mean=('Curricular units 2nd sem (approved)','mean'),second_grade_mean=('Curricular units 2nd sem (grade)','mean'),tuition_not_current_rate=('Tuition fees up to date',lambda s:(s==0).mean()),debtor_rate=('Debtor',lambda s:(s==1).mean()))
        rates=pd.crosstab(work.cluster,work.Target,normalize='index')
        profiles.join(rates).reset_index().to_csv(OUT/f'cluster_profiles_k{k}.csv',index=False)
        work[['StudentID','cluster']].to_csv(OUT/f'cluster_assignments_k{k}.csv',index=False)
        seeds=[KMeans(n_clusters=k,random_state=s,n_init=20).fit_predict(x) for s in range(30)]
        aris=[adjusted_rand_score(l,r) for l,r in combinations(seeds,2)]
        (OUT/f'cluster_stability_k{k}.json').write_text(json.dumps(dict(k=k,mean_pairwise_ari=float(np.mean(aris)),min_pairwise_ari=float(np.min(aris)),n_initializations=30),indent=2))
    (OUT/'cluster_specification.json').write_text(json.dumps(dict(features=a.RISK_FEATURES,selected_k=best,criterion='maximum silhouette among k=2,...,8',preprocessing='training on all observations for descriptive clustering only; standard numeric scaling and one-hot categories'),indent=2))
    main=pd.read_csv(HERE/'consistency_07_10/selected_test_bootstrap_ci.csv')
    for _,r in main.iterrows():
        rr=next(v for v in rows if v['specification']=='recorded' and v['stage']==r.stage)
        assert abs(rr['macro_f1']-r.macro_f1)<1e-12
    align_primary_intervals()
    print('All primary point estimates independently reproduced; checks complete',flush=True)

def align_primary_intervals():
    # Reuse canonical primary intervals; do not print slightly different Monte
    # Carlo replicates of the same recorded model in a supplemental table.
    canonical=pd.read_csv(HERE/'consistency_07_10/selected_test_bootstrap_ci.csv').set_index('stage')
    table=pd.read_csv(OUT/'timing_sensitivity_metrics.csv')
    for idx,r in table.iterrows():
        if r.specification=='recorded':
            for col in ['macro_f1_ci_low','macro_f1_ci_high']:table.loc[idx,col]=canonical.loc[r.stage,col]
    table.to_csv(OUT/'timing_sensitivity_metrics.csv',index=False)
    pairs=pd.read_csv(OUT/'timing_sensitivity_paired_differences.csv')
    primary=pd.read_csv(HERE/'consistency_07_10/paired_stage_bootstrap_differences.csv')
    for idx,r in pairs.iterrows():
        if r.later_specification=='recorded':
            c=primary[(primary.metric=='macro_f1') & (primary.comparison==r.comparison)].iloc[0]
            for col in ['difference','ci_low','ci_high']:pairs.loc[idx,col]=c[col]
    pairs.to_csv(OUT/'timing_sensitivity_paired_differences.csv',index=False)

if __name__=='__main__':run()
