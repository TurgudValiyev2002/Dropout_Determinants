"""Small independent smoke test; not a full analysis reproduction."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import critical_analysis as a
import pandas as pd
from sklearn.metrics import f1_score
assert a.DF.groupby('Split').size().to_dict()=={'test':442,'train':3097,'validation':885}
assert a.TARGET not in a.META['stage_feature_sets']['C_second_semester']
saved=pd.read_csv(ROOT/'consistency_07_10/selected_test_bootstrap_ci.csv').set_index('stage')
train=a.DF[a.DF.Split=='train'];test=a.DF[a.DF.Split=='test']
for stage,params in a.SELECTED_PARAMS.items():
    xtr,xt=a.prep(train,test,a.META['stage_feature_sets'][stage])
    model=a.model(params);model.set_params(n_jobs=4);model.fit(xtr,train.y)
    score=f1_score(test.y,model.predict(xt),average='macro')
    assert abs(score-saved.loc[stage,'macro_f1'])<1e-8,(stage,score)
    print(stage,score)
print('Input counts and three primary model point estimates verified.')
