"""
Test script for the new SimpleState implementation
"""

from main import SQLiRLTrainer
from simple_state import SimpleStateManager
import numpy as np

def test_simple_state_integration():
    """Test SimpleState integration with the full system"""
    
    print("🧪 TESTING SIMPLE STATE INTEGRATION")
    print("=" * 60)
    
    # Test configuration
    config = {
        'target_url': 'https://www.zixem.altervista.org/SQLi/level1.php',
        'parameter': 'id',
        'injection_point': '1',
        'method': 'GET',
        'num_episodes': 1,        # Just 1 episode
        'max_steps_per_episode': 5,  # Just 5 steps
        'learning_rate': 0.001,
        'initial_temperature': 1.0,  # Lower temperature for more deterministic behavior
        'save_frequency': 1,
        'log_frequency': 1,
        'debug_mode': True,
        'debug_frequency': 1
    }
    
    try:
        print("🚀 Step 1: Initialize Trainer with SimpleState")
        trainer = SQLiRLTrainer(config)
        
        print(f"✅ Trainer initialized!")
        print(f"   • State size: {trainer.env.get_state_size()}")
        print(f"   • Action size: {trainer.env.get_action_size()}")
        print(f"   • Using SimpleStateManager: {type(trainer.env.state_manager).__name__}")
        
        print(f"\n🔄 Step 2: Test Environment Reset")
        initial_state = trainer.env.reset()
        print(f"   • Initial state shape: {initial_state.shape}")
        print(f"   • Initial state range: [{initial_state.min():.3f}, {initial_state.max():.3f}]")
        print(f"   • Initial payload: '{trainer.env.current_payload}'")
        
        print(f"\n🎯 Step 3: Test Action Selection")
        action = trainer.agent.select_token(initial_state)
        token_name = trainer.env.gen_action.get_token_name(action)
        print(f"   • Selected action: {action}")
        print(f"   • Token name: '{token_name}'")
        
        # Get Q-values for analysis
        q_values = trainer.agent.get_q_values(initial_state)
        print(f"   • Q-values range: [{q_values.min():.4f}, {q_values.max():.4f}]")
        print(f"   • Q-values std: {q_values.std():.4f}")
        
        print(f"\n🌐 Step 4: Test Environment Step")
        next_state, reward, done, info = trainer.env.step(action)
        
        print(f"   • Step completed!")
        print(f"   • New state shape: {next_state.shape}")
        print(f"   • New state range: [{next_state.min():.3f}, {next_state.max():.3f}]")
        print(f"   • Reward: {reward}")
        print(f"   • Done: {done}")
        print(f"   • Updated payload: '{info['payload']}'")
        print(f"   • Response status: {info['response_status']}")
        
        print(f"\n🔍 Step 5: Debug State Features")
        debug_info = trainer.env.state_manager.debug_state(next_state)
        
        print(f"   📊 Payload Features (first 5):")
        payload_features = list(debug_info['payload_features'].items())[:5]
        for name, value in payload_features:
            print(f"     • {name}: {value:.3f}")
        
        print(f"   📡 Response Features (first 5):")
        response_features = list(debug_info['response_features'].items())[:5]
        for name, value in response_features:
            print(f"     • {name}: {value:.3f}")
        
        print(f"   🛡️ WAF Features (first 5):")
        waf_features = list(debug_info['waf_features'].items())[:5]
        for name, value in waf_features:
            print(f"     • {name}: {value:.3f}")
        
        print(f"   📈 Progress Features:")
        progress_features = list(debug_info['progress_features'].items())
        for name, value in progress_features:
            print(f"     • {name}: {value:.3f}")
        
        print(f"\n🎓 Step 6: Test Multiple Steps")
        for step in range(2, 4):  # Steps 2-3
            action = trainer.agent.select_token(next_state)
            token_name = trainer.env.gen_action.get_token_name(action)
            next_state, reward, done, info = trainer.env.step(action)
            
            print(f"   Step {step}: '{token_name}' → Payload: '{info['payload']}' → Reward: {reward:.2f}")
            
            if done:
                print(f"   Episode ended at step {step}")
                break
        
        print(f"\n✅ SIMPLE STATE INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SIMPLE STATE INTEGRATION TEST FAILED!")
        print(f"   • Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_manager_directly():
    """Test SimpleStateManager directly"""
    
    print(f"\n🔧 TESTING STATE MANAGER DIRECTLY")
    print("=" * 60)
    
    try:
        state_manager = SimpleStateManager()
        
        print(f"📊 Test 1: Empty State")
        empty_state = state_manager.build_state("", step_count=0, max_steps=50)
        print(f"   • Shape: {empty_state.shape}")
        print(f"   • Range: [{empty_state.min():.3f}, {empty_state.max():.3f}]")
        print(f"   • Non-zero features: {np.count_nonzero(empty_state)}")
        
        print(f"\n📊 Test 2: Simple Payload")
        simple_state = state_manager.build_state("SELECT", step_count=1, max_steps=50, reward=0.1)
        print(f"   • Shape: {simple_state.shape}")
        print(f"   • Range: [{simple_state.min():.3f}, {simple_state.max():.3f}]")
        print(f"   • Non-zero features: {np.count_nonzero(simple_state)}")
        
        print(f"\n📊 Test 3: Complex Payload with Response")
        test_response = {
            'status_code': 200,
            'content': 'Unknown column \'test\' in \'where clause\'',
            'content_length': 1000,
            'response_time': 0.5
        }
        
        complex_state = state_manager.build_state(
            current_payload="SELECT * FROM users WHERE id=1 UNION SELECT 1,2,3--",
            response=test_response,
            is_blocked=False,
            bypass_applied=True,
            bypass_method='comment_insertion',
            step_count=5,
            max_steps=50,
            reward=0.7
        )
        
        print(f"   • Shape: {complex_state.shape}")
        print(f"   • Range: [{complex_state.min():.3f}, {complex_state.max():.3f}]")
        print(f"   • Non-zero features: {np.count_nonzero(complex_state)}")
        
        # Debug the complex state
        debug_info = state_manager.debug_state(complex_state)
        print(f"   • Payload features with values > 0:")
        for name, value in debug_info['payload_features'].items():
            if value > 0:
                print(f"     - {name}: {value:.3f}")
        
        print(f"   • Response features with values > 0:")
        for name, value in debug_info['response_features'].items():
            if value > 0:
                print(f"     - {name}: {value:.3f}")
        
        print(f"\n✅ STATE MANAGER DIRECT TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ STATE MANAGER DIRECT TEST FAILED!")
        print(f"   • Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("🎯 COMPREHENSIVE SIMPLE STATE TEST")
    print("=" * 80)
    
    # Test state manager directly first
    direct_test = test_state_manager_directly()
    
    if direct_test:
        # Test full integration
        integration_test = test_simple_state_integration()
        
        if integration_test:
            print(f"\n🎉 ALL SIMPLE STATE TESTS PASSED!")
            print(f"   • SimpleStateManager working correctly")
            print(f"   • Integration with RL system successful")
            print(f"   • Ready for training with new state representation")
        else:
            print(f"\n⚠️ INTEGRATION TEST FAILED")
    else:
        print(f"\n⚠️ DIRECT TEST FAILED")

if __name__ == "__main__":
    main()
