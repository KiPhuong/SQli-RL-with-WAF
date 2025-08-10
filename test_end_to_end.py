"""
End-to-end test to verify the complete system works
"""

from main import SQLiRLTrainer
import numpy as np

def test_system_integration():
    """Test complete system integration"""
    
    print("🧪 END-TO-END SYSTEM TEST")
    print("=" * 60)
    
    # Test configuration
    config = {
        'target_url': 'https://www.zixem.altervista.org/SQLi/level1.php',
        'parameter': 'id',
        'injection_point': '1',
        'method': 'GET',
        'num_episodes': 2,        # Small test
        'max_steps_per_episode': 5,  # Small test
        'learning_rate': 0.001,
        'initial_temperature': 2.0,
        'save_frequency': 1,
        'log_frequency': 1,
        'debug_mode': True,
        'debug_frequency': 1
    }
    
    try:
        print("🚀 Step 1: Initialize Trainer")
        trainer = SQLiRLTrainer(config)
        
        print(f"✅ Trainer initialized successfully!")
        print(f"   • State size: {trainer.env.get_state_size()}")
        print(f"   • Action size: {trainer.env.get_action_size()}")
        print(f"   • Vocabulary: {trainer.env.gen_action.get_vocab_size()} tokens")
        
        print(f"\n🧠 Step 2: Test Agent Components")
        
        # Test state creation
        initial_state = trainer.env.reset()
        print(f"   • Initial state shape: {initial_state.shape}")
        print(f"   • Initial state sample: {initial_state[:5]}")
        
        # Test action selection
        action = trainer.agent.select_token(initial_state)
        token_name = trainer.env.gen_action.get_token_name(action)
        print(f"   • Selected action: {action} ('{token_name}')")
        
        # Test Q-values
        q_values = trainer.agent.get_q_values(initial_state)
        print(f"   • Q-values shape: {q_values.shape}")
        print(f"   • Q-values sample: {q_values[:5]}")
        
        print(f"\n🌐 Step 3: Test Environment Step")
        
        # Test environment step
        next_state, reward, done, info = trainer.env.step(action)
        print(f"   • Step completed successfully!")
        print(f"   • Reward: {reward}")
        print(f"   • Done: {done}")
        print(f"   • Payload: '{info['payload']}'")
        print(f"   • Final URL: {info['final_url']}")
        print(f"   • Response status: {info['response_status']}")
        
        print(f"\n🎓 Step 4: Test Training Loop")
        
        # Run minimal training
        trainer.train()
        
        print(f"\n✅ END-TO-END TEST PASSED!")
        print(f"   • All components working correctly")
        print(f"   • System ready for full training")
        
        return True
        
    except Exception as e:
        print(f"\n❌ END-TO-END TEST FAILED!")
        print(f"   • Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components separately"""
    
    print(f"\n🔧 COMPONENT TESTS")
    print("=" * 60)
    
    try:
        # Test GenAction
        print("📝 Testing GenAction...")
        from gen_action import GenAction
        gen_action = GenAction()
        
        vocab_size = gen_action.get_vocab_size()
        print(f"   ✅ Vocabulary size: {vocab_size}")
        
        initial_state = gen_action.create_initial_state()
        print(f"   ✅ Initial state: {initial_state.shape}")
        
        # Test BypassWAF
        print("🛡️ Testing BypassWAF...")
        from bypass_waf import BypassWAF
        bypass_waf = BypassWAF()
        
        test_token = "SELECT"
        should_bypass = bypass_waf.should_bypass_token(test_token)
        print(f"   ✅ Should bypass '{test_token}': {should_bypass}")
        
        if should_bypass:
            bypass_result = bypass_waf.apply_bypass_to_token(test_token)
            print(f"   ✅ Bypass result: {bypass_result}")
        
        # Test Agent
        print("🧠 Testing Agent...")
        from agent import SQLiRLAgent
        
        state_size = len(initial_state)
        action_size = vocab_size
        agent = SQLiRLAgent(state_size, action_size)
        
        action = agent.select_token(initial_state)
        print(f"   ✅ Selected action: {action}")
        
        q_values = agent.get_q_values(initial_state)
        print(f"   ✅ Q-values computed: {q_values.shape}")
        
        print(f"\n✅ ALL COMPONENT TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ COMPONENT TEST FAILED!")
        print(f"   • Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("🎯 COMPREHENSIVE SYSTEM TEST")
    print("=" * 80)
    
    # Test individual components first
    component_test = test_individual_components()
    
    if component_test:
        # Test full system integration
        integration_test = test_system_integration()
        
        if integration_test:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"   • System is ready for production use")
            print(f"   • You can now run: python run_sqli_rl.py --url <target>")
        else:
            print(f"\n⚠️ INTEGRATION TEST FAILED")
            print(f"   • Components work individually but not together")
    else:
        print(f"\n⚠️ COMPONENT TESTS FAILED")
        print(f"   • Fix individual components first")

if __name__ == "__main__":
    main()
