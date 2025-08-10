"""
Test script for the new injection logic
"""

from env import SQLiEnvironment

def test_injection_patterns():
    """Test different injection patterns"""
    
    print("🧪 TESTING INJECTION LOGIC")
    print("=" * 60)
    
    # Create environment
    env = SQLiEnvironment(
        target_url="https://example.com/vuln.php?id=1",
        parameter="id",
        injection_point="1",
        method="GET"
    )
    
    # Test cases
    test_cases = [
        # (payload, expected_pattern, description)
        ("", "?id=1", "Empty payload"),
        ("UNION SELECT 1,2,3", "?id=1 UNION SELECT 1,2,3", "Union-based"),
        ("' UNION SELECT 1,2,3--", "?id=1' UNION SELECT 1,2,3--", "Union with quotes"),
        ("AND 1=1", "?id=1 AND 1=1", "Boolean AND"),
        ("OR 1=1", "?id=1 OR 1=1", "Boolean OR"),
        ("' AND '1'='1", "?id=1' AND '1'='1", "String boolean"),
        ("AND EXTRACTVALUE(1,CONCAT(0x7e,VERSION()))", "?id=1 AND AND EXTRACTVALUE(1,CONCAT(0x7e,VERSION()))", "Error-based"),
        ("EXTRACTVALUE(1,CONCAT(0x7e,VERSION()))", "?id=1 AND EXTRACTVALUE(1,CONCAT(0x7e,VERSION()))", "Error-based (auto AND)"),
        ("AND SLEEP(5)", "?id=1 AND AND SLEEP(5)", "Time-based"),
        ("SLEEP(5)", "?id=1 AND SLEEP(5)", "Time-based (auto AND)"),
        ("'", "?id=1' '", "Single quote"),
        ("--", "?id=1 --", "Comment"),
        ("/**/", "?id=1 /**/", "Block comment"),
        ("+ 1", "?id=1 + 1", "Arithmetic"),
    ]
    
    print("🔍 Testing injection patterns:")
    print("-" * 60)
    
    for payload, expected_pattern, description in test_cases:
        try:
            # Test injection value building
            injection_value = env._build_injection_value(payload)
            
            # Test full URL building  
            full_url = env._build_injection_url(payload)
            
            print(f"📝 {description}:")
            print(f"   Input:    '{payload}'")
            print(f"   Value:    '{injection_value}'")
            print(f"   Full URL: {full_url}")
            
            # Check if it matches expected pattern
            if expected_pattern in full_url:
                print(f"   Status:   ✅ PASS")
            else:
                print(f"   Status:   ❌ FAIL (expected: {expected_pattern})")
            print()
            
        except Exception as e:
            print(f"   Status:   ❌ ERROR: {e}")
            print()

def test_different_targets():
    """Test with different target URL formats"""
    
    print("🌐 TESTING DIFFERENT TARGET FORMATS")
    print("=" * 60)
    
    targets = [
        ("https://example.com/vuln.php", "id", "1"),
        ("https://example.com/vuln.php?id=1", "id", "1"), 
        ("https://example.com/vuln.php?page=home&id=1", "id", "1"),
        ("https://example.com/vuln.php?id=1&debug=true", "id", "1"),
    ]
    
    test_payload = "UNION SELECT 1,2,3"
    
    for target_url, param, injection_point in targets:
        print(f"🎯 Target: {target_url}")
        print(f"   Parameter: {param}")
        print(f"   Injection Point: {injection_point}")
        
        try:
            env = SQLiEnvironment(
                target_url=target_url,
                parameter=param,
                injection_point=injection_point,
                method="GET"
            )
            
            final_url = env._build_injection_url(test_payload)
            print(f"   Result: {final_url}")
            print(f"   Status: ✅ SUCCESS")
            
        except Exception as e:
            print(f"   Status: ❌ ERROR: {e}")
        
        print()

def test_post_method():
    """Test POST method injection"""
    
    print("📮 TESTING POST METHOD")
    print("=" * 60)
    
    env = SQLiEnvironment(
        target_url="https://example.com/vuln.php",
        parameter="id", 
        injection_point="1",
        method="POST"
    )
    
    test_payloads = [
        "",
        "UNION SELECT 1,2,3",
        "' OR '1'='1",
    ]
    
    for payload in test_payloads:
        try:
            final_url = env._build_injection_url(payload)
            print(f"Payload: '{payload}' → URL: {final_url}")
            
        except Exception as e:
            print(f"Payload: '{payload}' → ERROR: {e}")

def main():
    """Run all tests"""
    
    test_injection_patterns()
    test_different_targets()
    test_post_method()
    
    print("🎉 INJECTION LOGIC TESTING COMPLETED!")

if __name__ == "__main__":
    main()
