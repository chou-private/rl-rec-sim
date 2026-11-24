import unittest
import numpy as np
import torch

from experiments.yahoo_recording.core import (rating_counts, Marginal, Recording,
    recording_distribution, count_nll, random_nll, fit, metrics)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12);torch.set_num_threads(1)

    def test_counts_and_hand_likelihood(self):
        ratings=np.array([[0,1,1,-1,-1],[4,4,3,2,-1]])
        rows=np.array([[0,0,2],[0,3,2],[1,0,4]])
        a,r=rating_counts(ratings,rows)
        np.testing.assert_array_equal(a,[[1,2,0,0,0],[0,0,1,1,2]])
        np.testing.assert_array_equal(r,[[0,0,2,0,0],[0,0,0,0,1]])
        p=torch.full((1,5),.2);mu=torch.tensor([[.1,.2,.3,.4,.5]])
        dist=recording_distribution(p,mu)
        torch.testing.assert_close(dist,torch.tensor([[.02,.04,.06,.08,.1,.7]]))
        counts=torch.tensor([[1.,2.,0.,0.,0.]])
        expected=-(np.log(.02)+2*np.log(.04)+2*np.log(.7))/5
        self.assertAlmostEqual(count_nll(dist,counts,5).item(),expected,places=6)

    def test_global_fallback_and_probability_validity(self):
        base=torch.tensor([.01,.02,.03,.06,.2]);q=torch.randn(10,5)
        global_model=Recording(base);own=Recording(base,True)
        torch.testing.assert_close(global_model(q),own(q))
        dist=recording_distribution(torch.randn(10,5).softmax(-1),own(q))
        torch.testing.assert_close(dist.sum(-1),torch.ones(10))
        self.assertTrue((dist>0).all())

    def test_marginal_uses_star_labels_not_recorded_flags(self):
        prior=torch.tensor([.5,.2,.1,.1,.1]);model=Marginal(prior)
        p=model(torch.ones(2,5))
        torch.testing.assert_close(p,prior.expand(2,-1))
        counts=torch.tensor([[10.,0,0,0,0],[0.,0,0,0,10]])
        torch.testing.assert_close(random_nll(p,counts),-torch.tensor([.5,.1]).log())

    def test_global_fit_recovers_synthetic_mechanism(self):
        q=torch.zeros(4,5);p=torch.full((4,5),.2)
        true_mu=torch.tensor([.1,.2,.3,.4,.5])
        counts=(1000*p*true_mu).clone()
        model=Recording(torch.full((5,),.15))
        loss=lambda m:count_nll(recording_distribution(p,m(q)),counts,1000).mean()
        fit(model,loss,loss,epochs=300,patience=60,lr=.05)
        torch.testing.assert_close(model(q)[0],true_mu,atol=.003,rtol=0)
        report=metrics(p,model(q),counts,1000)
        self.assertLess(report['total_count_mae'].mean(),1)

    def test_epoch_zero_is_valid_fallback(self):
        model=Recording(torch.full((5,),.1),True)
        q=torch.ones(2,5);initial={k:v.clone() for k,v in model.state_dict().items()}
        result=fit(model,lambda m:-m(q).mean(),lambda m:torch.tensor(1.),epochs=5,patience=2)
        self.assertEqual(result['best_epoch'],0)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k])

    def test_conditional_network_can_learn_known_questionnaire_signal(self):
        q=torch.zeros(20,5);q[:10,0]=-1;q[10:,0]=1
        p=torch.full((20,5),.2);base=torch.full((5,),.15)
        mu=torch.sigmoid(torch.logit(base)+q[:,:1]*torch.tensor([[1.,.5,0.,-.5,-1.]]))
        counts=torch.round(1000*p*mu)
        model=Recording(base,True,8)
        loss=lambda m:count_nll(recording_distribution(p,m(q)),counts,1000).mean()
        start=loss(model).item()
        fit(model,loss,loss,epochs=150,patience=30,lr=.03)
        self.assertLess(loss(model).item(),start-.005)
        self.assertGreater(model(q)[-1,0].item(),model(q)[0,0].item())


if __name__=='__main__':unittest.main()
