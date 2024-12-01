import unittest
import random
import numpy as np

from ams.performance.hazard_effects import HazardEffect

class TestHazardEffect(unittest.TestCase):

    def setUp(self):
        """Set up a HazardEffect instance for testing."""
        hazard_data = [
            {
                "name": "Minor Damage",
                "probability": 0.161281,
                "degradation":   {
                    "1": [1, 1, 1],
                    "2": [1, 1, 1],
                    "3": [1, 1, 1],
                    "4": [1, 1, 1],
                }
            },
            {
                "name": "Moderate Damage",
                "probability": 0.068461,
                "degradation":   {
                    "1": [2, 2, 2],
                    "2": [2, 2, 2],
                    "3": [2, 2, 2],
                    "4": [2, 2, 2],
                }
            },{
                "name": "Severe Damage",
                "probability": 0.022321,
                "degradation":   {
                    "1": [3, 3, 3],
                    "2": [3, 3, 3],
                    "3": [3, 3, 3],
                    "4": [3, 3, 3],
                }
            },{
                "name": "Collapse",
                "probability": 0.000429,
                "degradation":   {
                    "1": [4, 4, 4],
                    "2": [4, 4, 4],
                    "3": [4, 4, 4],
                    "4": [4, 4, 4],
                }
            },
        ]

        self.hazard_effects = HazardEffect.set_hazard_effects(hazard_data)
                
        
    def test_initialization_probabilities(self):
        """Test initialization of HazardEffect attributes."""
        self.assertAlmostEqual(self.hazard_effects['No Damage'].probability, 0.747508, places=7)
        self.assertAlmostEqual(self.hazard_effects['Minor Damage'].probability, 0.161281, places=7)
        self.assertAlmostEqual(self.hazard_effects['Moderate Damage'].probability, 0.068461, places=7)
        self.assertAlmostEqual(self.hazard_effects['Severe Damage'].probability, 0.022321, places=7)
        self.assertAlmostEqual(self.hazard_effects['Collapse'].probability, 0.000429, places=7)
    
    def test_initialization_degradations(self):
        """Test initialization of HazardEffect attributes."""
        num_samples = 5
        current_state = 3
        np.random.seed(1)
        counts = [self.hazard_effects['Moderate Damage'].get_degradation(current_state) 
                  for i in range(num_samples)]
        
        self.assertAlmostEqual(np.mean(counts), 2, places=4)
        
    def test_set_effect(self):
        effect = {2: [0, 3, 3]}
        self.hazard_effects['Moderate Damage'].set_degradation(effect)
        self.assertEqual(self.hazard_effects['Moderate Damage'].degradation[2], effect[2])

    def test_random_sampling(self):
        """Test random sampling aligns with defined probabilities."""
        hazard_data = {
            "Damage": list(self.hazard_effects.keys()),
            "Probability": [self.hazard_effects[effect].probability for effect in self.hazard_effects],
        }

        # Normalize probabilities
        hazard_data['Probability'] = np.array(hazard_data['Probability']) / sum(hazard_data['Probability'])

        # Perform random sampling
        num_samples = 5000
        np.random.seed(1)
        samples = np.random.choice(
            hazard_data['Damage'],
            p=hazard_data['Probability'],
            size=num_samples
        )

        # Count occurrences
        sample_counts = {damage: (samples == damage).sum() for damage in hazard_data['Damage']}

        # Validate that the frequencies are close to the expected probabilities
        for damage in hazard_data['Damage']:
            expected_frequency = hazard_data['Probability'][hazard_data['Damage'].index(damage)]
            self.assertAlmostEqual(sample_counts[damage] / num_samples, expected_frequency, places=2)
            
    
if __name__ == '__main__':
    unittest.main()
