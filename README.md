# SQL Injection RL with WAF Bypass Framework

An advanced reinforcement learning framework for automated SQL injection testing with Web Application Firewall (WAF) bypass capabilities. This project combines deep reinforcement learning techniques with sophisticated payload generation and WAF detection/bypass methods to create an intelligent penetration testing tool.

## 🎯 Features

### Core Capabilities
- **Deep Reinforcement Learning**: DQN, Dueling DQN, and Attention-based DQN implementations
- **Advanced SQL Injection Testing**: Union-based, Boolean blind, Time-based, Error-based, and Stacked queries
- **WAF Detection & Bypass**: Support for 15+ WAF types including Cloudflare, AWS WAF, ModSecurity, F5, etc.
- **Intelligent Payload Generation**: Dynamic payload modification based on agent actions and environment feedback
- **Comprehensive Logging**: Structured logging with performance metrics, security events, and payload tracking

### Technical Components
- **RL Agent**: PyTorch-based neural networks with experience replay and target networks
- **Environment Simulation**: Realistic web application testing environment with HTTP request/response handling
- **State Management**: 100-dimensional state space with response analysis and feature extraction
- **Reward System**: Multi-faceted reward calculation promoting successful injection discovery and efficiency
- **Action Space**: 20+ predefined actions covering different injection techniques and bypass methods

### Guidelines & Compliance
- **OWASP Testing Guide**: Integration with OWASP penetration testing methodologies
- **PTES Framework**: Penetration Testing Execution Standard compliance
- **Rule Engine**: Condition-based decision making for automated testing guidance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- CUDA support (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sqli-rl-with-waf.git
   cd sqli-rl-with-waf
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

### Basic Usage

#### Training Mode
Train the RL agent on a target application:
```bash
python main.py train --target http://testphp.vulnweb.com --episodes 1000 --save-model models/my_model.pth
```

#### Testing Mode
Test a trained model:
```bash
python main.py test --model models/my_model.pth --target http://example.com --episodes 50
```

#### Interactive Mode
Explore the framework interactively:
```bash
python main.py interactive --model models/my_model.pth
```

### Configuration

The framework uses YAML configuration files. See `config.yaml` for all available options:

```yaml
# Agent configuration
agent:
  learning_rate: 0.001
  memory_size: 10000
  batch_size: 32
  network_type: "dqn"

# Environment configuration
environment:
  target_url: "http://testphp.vulnweb.com"
  max_steps_per_episode: 100
  timeout: 10

# Training configuration
training:
  episodes: 1000
  save_frequency: 100
  eval_frequency: 50
```

## 📊 Architecture Overview

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

## 🔬 Technical Details

### Reinforcement Learning Implementation

The framework implements several DQN variants:

- **Standard DQN**: Basic deep Q-network with experience replay
- **Dueling DQN**: Separate value and advantage streams
- **Attention DQN**: Attention mechanism for feature selection

```python
# Example: Creating and training an agent
agent = RLAgent(
    state_size=100,
    action_size=20,
    learning_rate=0.001,
    memory_size=10000
)

# Training loop
for episode in range(episodes):
    state = env.reset()
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
```

### WAF Detection

The system can detect and handle multiple WAF types:

```python
# WAF detection example
detector = WAFDetector()
waf_info = detector.detect_waf("http://example.com")
if waf_info['detected']:
    print(f"WAF detected: {waf_info['type']}")
    bypass_methods = detector.get_bypass_methods(waf_info['type'])
```

### Payload Generation

Dynamic payload generation based on agent actions:

```python
# Payload generation example
generator = PayloadGenerator(action_space, bypass_methods)
payload = generator.generate_payload(
    action=5,  # Union-based injection
    context={'waf_type': 'cloudflare', 'database': 'mysql'}
)
```

## 📈 Performance Metrics

The framework tracks comprehensive metrics:

- **Success Rate**: Percentage of episodes resulting in successful injection
- **WAF Bypass Rate**: Success rate of WAF bypass attempts
- **Average Episode Length**: Number of steps to completion
- **Payload Effectiveness**: Success rates by payload type
- **Response Time Analysis**: Performance impact measurement

## 🛡️ Security Considerations

### Ethical Usage
This framework is designed for:
- ✅ Authorized penetration testing
- ✅ Security research and education
- ✅ Vulnerability assessment of owned systems
- ✅ Red team exercises with proper authorization

### Disclaimer
⚠️ **Important**: This tool is for educational and authorized testing purposes only. Users are responsible for ensuring they have proper authorization before testing any systems. Unauthorized testing of systems you do not own is illegal and unethical.

### Safety Features
- Built-in request rate limiting
- Configurable target restrictions
- Comprehensive audit logging
- Payload sanitization options

## 🔧 Advanced Configuration

### Custom Neural Networks
```yaml
agent:
  network_type: "custom"
  hidden_sizes: [512, 256, 128]
  activation: "relu"
  dropout: 0.2
```

### WAF Configuration
```yaml
waf:
  detection_enabled: true
  bypass_enabled: true
  max_bypass_attempts: 5
  confidence_threshold: 0.7
```

### Payload Customization
```yaml
payloads:
  max_payload_length: 1000
  encoding_methods: ["url", "html", "unicode"]
  custom_payloads_file: "custom_payloads.json"
```

## 📝 Logging and Monitoring

The framework provides comprehensive logging:

### Log Types
- **Application Logs**: General framework operation
- **Payload Logs**: Detailed payload attempt information (JSON format)
- **Performance Logs**: Training and testing metrics
- **Security Events**: Injection discoveries and WAF detections

### Log Analysis
```python
# Accessing logs programmatically
logger = get_logger()
stats = logger.get_log_statistics()
print(f"Total requests: {stats['total_requests']}")

# Export logs for analysis
logs = logger.export_logs(
    start_time=datetime(2024, 1, 1),
    event_types=['injection_success', 'waf_detection']
)
```

## 🧪 Testing and Validation

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_agent.py
pytest tests/test_environment.py
pytest tests/test_waf.py
```

### Integration Tests
```bash
# Test against vulnerable applications
python scripts/test_integration.py

# Validate WAF detection
python scripts/test_waf_detection.py
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📚 Research and Papers

This framework is based on research in:
- Deep Reinforcement Learning for Security Testing
- Automated Vulnerability Discovery
- WAF Bypass Techniques
- Adversarial Machine Learning in Cybersecurity

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OWASP Foundation for testing methodologies
- PyTorch team for the deep learning framework
- Security research community for vulnerability disclosure practices
- Academic researchers in adversarial ML and automated security testing

## 📞 Support and Contact

- **Issues**: Please use the GitHub issues tracker
- **Discussions**: Join our community discussions
- **Email**: your.email@example.com
- **Documentation**: See our [Wiki](https://github.com/yourusername/sqli-rl-with-waf/wiki)

## 🔮 Roadmap

### Version 1.1 (Planned)
- [ ] Support for NoSQL injection testing
- [ ] GraphQL endpoint testing
- [ ] Advanced WAF fingerprinting
- [ ] Multi-threaded testing capabilities

### Version 1.2 (Future)
- [ ] Web UI for framework management
- [ ] Report generation and visualization
- [ ] Integration with popular security tools
- [ ] Cloud deployment options

---

**Remember**: Always test responsibly and only on systems you own or have explicit permission to test. Happy hunting! 🕵️‍♂️