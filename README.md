# SQL Injection RL with WAF Bypass Framework

### Project Structure
```
SQli-RL-with-WAF/
├── src/
│   ├── agent/                 # RL agent components
│   │   ├── rl_agent.py       # Main RL agent implementation
│   │   ├── neural_network.py # Neural network architectures
│   │   ├── boltzmann_exploration.py # Exploration strategies
│   │   └── action_space.py   # Action definitions
│   ├── environment/           # Testing environment
│   │   ├── pentest_env.py    # Main environment
│   │   ├── state_manager.py  # State representation
│   │   └── reward_calculator.py # Reward system
│   ├── waf/                  # WAF detection and bypass
│   │   ├── waf_detector.py   # WAF detection logic
│   │   └── bypass_methods.py # Bypass techniques
│   ├── payloads/             # Payload management
│   │   ├── payload_generator.py # Dynamic payload creation
│   │   └── payload_catalog.py # Payload database
│   ├── guidelines/           # Testing guidelines
│   │   ├── catalog_parser.py # OWASP/PTES integration
│   │   └── rule_engine.py    # Rule-based decisions
│   └── utils/                # Utilities
│       ├── network_utils.py  # HTTP operations
│       └── logging_utils.py  # Comprehensive logging
├── data/                     # Data files
│   ├── payload_catalog.json  # Payload database
│   └── pentest_catalog.json  # Guidelines catalog
├── logs/                     # Log files
├── models/                   # Trained models
├── config.yaml              # Configuration
├── main.py                  # Main application
└── requirements.txt         # Dependencies
```

### Component Interaction Flow

1. **Agent Decision**: RL agent selects action based on current state
2. **Payload Generation**: Action triggers payload modification/generation
3. **WAF Detection**: System detects presence and type of WAF
4. **Bypass Application**: Appropriate bypass methods applied to payload
5. **Request Execution**: HTTP request sent with modified payload
6. **Response Analysis**: Response analyzed for injection indicators
7. **State Update**: Environment state updated with new information
8. **Reward Calculation**: Reward computed based on success/failure
9. **Learning**: Agent updates neural network based on experience

