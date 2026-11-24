import unittest
import tempfile
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments.yahoo_downstream.core import state_vector, leave, double_target, evaluate, Replay, train


class DownstreamTests(unittest.TestCase):
    def test_shared_state_and_short_window(self):
        emb=np.arange(12,dtype=np.float32).reshape(6,2)
        np.testing.assert_array_equal(state_vector([],[],emb),np.zeros(3))
        np.testing.assert_allclose(state_vector([1,2],[0,1],emb),[3,4,.5])
        distance=np.full((6,6),200.0);distance[2,1]=20
        self.assertTrue(leave([1],2,distance,120,3))
        self.assertFalse(leave([],2,distance,120,3))
        self.assertFalse(leave([1,3,4,5],2,distance,120,3))
        shared={'item':emb, 'user':np.array([[1,2],[3,4]],dtype=np.float32)}
        self.assertFalse(np.array_equal(state_vector([],[],shared,user=0),state_vector([],[],shared,user=1)))

    def test_double_dqn_mask_and_terminal_target(self):
        online=torch.tensor([[100.,2.,3.],[1.,2.,3.]])
        target=torch.tensor([[99.,20.,30.],[10.,20.,30.]])
        mask=torch.tensor([[False,True,True],[False,False,False]])
        result=double_target(online,target,mask,torch.tensor([1.,2.]),torch.tensor([False,True]),.9)
        torch.testing.assert_close(result,torch.tensor([28.,2.]))

    def test_replay_copies_masks(self):
        replay=Replay(2,2,3)
        mask=np.ones(3,bool)
        replay.add([1,2],0,.5,[3,4],False,mask)
        mask[:]=False
        self.assertTrue(replay.mask[0].all())

    def test_common_evaluation_and_no_repeat(self):
        pred=np.array([[5.,4.,3.,2.]])
        gt=np.array([[1.,2.,3.,4.]])
        distance=np.zeros((4,4))
        args=SimpleNamespace(horizon=3,max_turn=4,window=3,leave_threshold=120,device='cpu')
        emb=np.zeros((4,2),np.float32)
        fixed=evaluate(None,pred,gt,distance,[0],[123],[2],emb,args,'fixed')
        natural=evaluate(None,pred,gt,distance,[0],[123],[2],emb,args,'natural')
        self.assertEqual(fixed['users'][0]['actions'],[0,1,2])
        self.assertEqual(fixed['summary']['mean_reward'],6)
        self.assertEqual(natural['users'][0]['actions'],[0,1])
        self.assertEqual(natural['summary']['mean_reward'],3)
        self.assertEqual(fixed['users'][0]['user_id'],123)

    def test_test_rewards_do_not_affect_training_or_selection(self):
        torch.set_num_threads(1)
        pred=np.array([[2.,3.,4.,5.]]*3)
        gt=pred.copy();changed=gt.copy();changed[2]=1
        distance=np.full((4,4),200.)
        pools={'calibration':np.array([0]),'validation':np.array([1]),'test':np.array([2])}
        emb={'item':np.arange(8,dtype=np.float32).reshape(4,2)/10,
             'user':np.arange(6,dtype=np.float32).reshape(3,2)/10}
        args=SimpleNamespace(device='cpu',hidden=4,lr=.001,capacity=30,epochs=2,steps_per_epoch=10,
                             window=3,epsilon=.1,learning_starts=4,batch_size=4,gamma=.9,
                             target_update=5,max_turn=3,horizon=2,leave_threshold=120)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir()
            m1=train(pred,gt,distance,pools,np.arange(1,4),np.zeros(3,int),emb,args,42,a)
            m2=train(pred,changed,distance,pools,np.arange(1,4),np.zeros(3,int),emb,args,42,b)
            for key,value in m1.state_dict().items():
                torch.testing.assert_close(value,m2.state_dict()[key],rtol=0,atol=0)
            self.assertEqual((a/'selection.json').read_text(),(b/'selection.json').read_text())


if __name__=='__main__': unittest.main()
