import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.yahoo_downstream.core import state_vector, state_dim, QNetwork, train
from experiments.yahoo_downstream.questionnaire_state import questionnaire_features, with_questionnaire
from experiments.yahoo_downstream.questionnaire_reward import permutation_within_pools


class StateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.shared={'user':np.arange(6,dtype=np.float32).reshape(3,2)/10,
                     'item':np.arange(8,dtype=np.float32).reshape(4,2)/10}

    def test_absolute_answers_preserved(self):
        f=questionnaire_features(np.array([[1,2,3,4,5],[5,5,5,5,5],[1,1,1,1,1]]))
        np.testing.assert_array_equal(f[0],[-1,-.5,0,.5,1])
        np.testing.assert_array_equal(f[1],np.ones(5))
        np.testing.assert_array_equal(f[2],-np.ones(5))
        with self.assertRaises(ValueError):questionnaire_features(np.zeros((3,5)))

    def test_same_dimension_fixed_user_features_and_unchanged_history(self):
        own=questionnaire_features(np.array([[1,2,3,4,5],[5,4,3,2,1],[5,5,5,5,5]]))
        for features in [np.zeros_like(own),own,own[[2,0,1]]]:
            emb=with_questionnaire(self.shared,features)
            self.assertEqual(state_dim(emb),state_dim(self.shared)+5)
            for user in range(3):
                for actions,rewards in [([],[]),([0,2],[.2,.7])]:
                    s=state_vector(actions,rewards,emb,user=user)
                    np.testing.assert_array_equal(s[-5:],features[user])
                    np.testing.assert_array_equal(s[:-5],state_vector(actions,rewards,self.shared,user=user))
        self.assertNotIn('questionnaire',self.shared)

    def test_shuffle_keeps_vectors_and_pools(self):
        pools={'calibration':np.arange(10),'validation':np.arange(10,15),'test':np.arange(15,20)}
        p=permutation_within_pools(20,pools,20261124)
        np.testing.assert_array_equal(p,permutation_within_pools(20,pools,20261124))
        for ids in pools.values():np.testing.assert_array_equal(np.sort(p[ids]),ids)

    def test_features_reach_learning_and_test_rewards_do_not(self):
        args=SimpleNamespace(device='cpu',hidden=8,lr=.01,capacity=30,epochs=2,steps_per_epoch=20,
             window=3,epsilon=.1,learning_starts=4,batch_size=4,gamma=.9,target_update=5,
             max_turn=3,horizon=2,leave_threshold=120)
        pred=np.array([[2.,3.,4.,5.]]*3);gt=pred.copy();changed=gt.copy();changed[2]=1
        pools={'calibration':np.array([0]),'validation':np.array([1]),'test':np.array([2])}
        distance=np.full((4,4),200.)
        own=with_questionnaire(self.shared,np.ones((3,5),np.float32))
        zero=with_questionnaire(self.shared,np.zeros((3,5),np.float32))
        torch.manual_seed(42)
        initial=QNetwork(state_dim(own),4,8).state_dict()['net.0.weight'].clone()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            paths=[Path(tmp)/str(i) for i in range(3)]
            for p in paths:p.mkdir()
            m=train(pred,gt,distance,pools,np.arange(1,4),np.zeros(3,int),own,args,42,paths[0])
            m2=train(pred,changed,distance,pools,np.arange(1,4),np.zeros(3,int),own,args,42,paths[1])
            mz=train(pred,gt,distance,pools,np.arange(1,4),np.zeros(3,int),zero,args,42,paths[2])
            for k,v in m.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)
            self.assertEqual((paths[0]/'selection.json').read_text(),(paths[1]/'selection.json').read_text())
            torch.testing.assert_close(mz.state_dict()['net.0.weight'][:,-5:],initial[:,-5:],rtol=0,atol=0)
            self.assertGreater((m.state_dict()['net.0.weight'][:,-5:]-initial[:,-5:]).abs().sum().item(),0)


if __name__=='__main__':unittest.main()
