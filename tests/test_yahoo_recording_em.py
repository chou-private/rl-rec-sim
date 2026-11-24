import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.yahoo_recording.core import Recording
from experiments.yahoo_recording.em import validate_mapping, load_probabilities
from experiments.yahoo_em.model import CategoricalDeepFM
from experiments.yahoo_em.run import e_step


class RecordingEMTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(5);torch.set_num_threads(1)

    def test_donor_mapping_must_remain_inside_splits(self):
        splits={'calibration':[1,2],'validation':[3,4],'test':[5,6]}
        valid={'user_ids':[1,2,3,4,5,6],'donor_ids':[2,1,4,3,6,5]}
        self.assertEqual(validate_mapping(valid,splits)[1],2)
        with self.assertRaises(ValueError):
            validate_mapping({'user_ids':valid['user_ids'],'donor_ids':[3,2,1,4,5,6]},splits)

    def test_global_fallback_and_shuffled_input_loading(self):
        base=torch.tensor([.01,.02,.03,.06,.2])
        global_model=Recording(base)
        own=Recording(base,True,8)
        with torch.no_grad():
            own.correction[0].weight.fill_(.1);own.correction[0].bias.zero_()
            own.correction[-1].weight.fill_(.2)
        fallback=Recording(base,True,8)
        q=np.array([[1]*5,[-1]*5],dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'seed_7';d.mkdir()
            for name,model in [('global',global_model),('own',own),('shuffled',fallback)]:
                folder=root if name=='global' else d
                torch.save(model.state_dict(),folder/f'{name}.pt')
                (folder/f'{name}_selection.json').write_text(json.dumps({'best_epoch':0}))
            tables,audit=load_probabilities(root,7,8,q,q[::-1].copy())
            np.testing.assert_allclose(tables['global'],tables['shuffled'],atol=1e-7)
            self.assertGreater(tables['own'][0,0],tables['own'][1,0])
            # Replacing only shuffled checkpoint tests that its donor inputs,
            # rather than own Q, are actually used at inference.
            torch.save(own.state_dict(),d/'shuffled.pt')
            tables,_=load_probabilities(root,7,8,q,q[::-1].copy())
            np.testing.assert_array_equal(tables['own'][::-1],tables['shuffled'])

    def test_per_user_table_e_step_and_global_equivalence(self):
        n=7;ratings=np.full((n,3),-1,dtype=int);ratings[0,0]=4
        model=CategoricalDeepFM(n,3,4,8)
        base=np.array([.01,.02,.03,.06,.2],dtype=np.float32)
        global_targets=e_step(model,ratings,np.zeros(n,int),base[None],5,'cpu')
        tables=np.tile(base,(n,1))
        per_user=e_step(model,ratings,np.arange(n),tables,5,'cpu')
        np.testing.assert_array_equal(global_targets,per_user)
        tables[-1,4]=.8
        changed=e_step(model,ratings,np.arange(n),tables,5,'cpu')
        np.testing.assert_array_equal(changed[:-3],per_user[:-3])
        self.assertTrue(np.all(changed[-3:,4]<per_user[-3:,4]))
        np.testing.assert_array_equal(changed[0],[0,0,0,0,1])


if __name__=='__main__':unittest.main()
