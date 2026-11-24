import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.yahoo_downstream.questionnaire_reward import (
    centered_scores, permutation_within_pools, auxiliary_from_probabilities)
from experiments.yahoo_downstream.core import train


class RewardTests(unittest.TestCase):
    def test_flat_zero_and_centering(self):
        scores=centered_scores(np.array([[3,3,3,3,3],[1,2,3,4,5]]))
        np.testing.assert_array_equal(scores[0],np.zeros(5))
        np.testing.assert_allclose(scores[1],[-.5,-.25,0,.25,.5])
        np.testing.assert_allclose(scores.sum(1),0,atol=1e-7)

    def test_probability_expectation_gate(self):
        p=np.array([[.1,.1,.2,.3,.3]]*2)
        s=np.array([[-.5,-.25,0,.25,.5]]*2)
        np.testing.assert_allclose(auxiliary_from_probabilities(p,s,np.array([3.6,2.9]),3),[.15,0])

    def test_permutation_preserves_answer_vectors_and_split(self):
        pools={'calibration':np.arange(10),'validation':np.arange(10,15),'test':np.arange(15,20)}
        a=permutation_within_pools(20,pools,123)
        np.testing.assert_array_equal(a,permutation_within_pools(20,pools,123))
        self.assertTrue((a!=np.arange(20)).any())
        for v in pools.values():np.testing.assert_array_equal(np.sort(a[v]),v)

    def test_zero_alpha_reproduces_original_training(self):
        torch.set_num_threads(1)
        pred=np.array([[2.,3.,4.,5.]]*3)
        gt=pred.copy();dist=np.full((4,4),200.)
        pools={'calibration':np.array([0]),'validation':np.array([1]),'test':np.array([2])}
        emb={'item':np.arange(8,dtype=np.float32).reshape(4,2)/10,
             'user':np.arange(6,dtype=np.float32).reshape(3,2)/10}
        args=SimpleNamespace(device='cpu',hidden=4,lr=.001,capacity=30,epochs=2,steps_per_epoch=10,
              window=3,epsilon=.1,learning_starts=4,batch_size=4,gamma=.9,target_update=5,max_turn=3,
              horizon=2,leave_threshold=120)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir()
            m1=train(pred,gt,dist,pools,np.arange(1,4),np.zeros(3,int),emb,args,42,a)
            m2=train(pred,gt,dist,pools,np.arange(1,4),np.zeros(3,int),emb,args,42,b,
                     auxiliary=np.full_like(pred,.5),alpha=0)
            for k,v in m1.state_dict().items():torch.testing.assert_close(v,m2.state_dict()[k],rtol=0,atol=0)
            self.assertEqual((a/'selection.json').read_text(),(b/'selection.json').read_text())


if __name__=='__main__':unittest.main()
