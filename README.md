# 🧬 GENESIS v3 - Universal Code Evolution Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A code evolution engine that automatically optimizes Python code using genetic algorithms.

## ✨ Features

- 🎯 **Automatic Optimization**: Optimizes functions and methods using genetic algorithms
- 📊 **Performance Analysis**: Detailed comparison between original and evolved code
- 🔄 **Smart Mutations**: AST-based meaningful code mutations
- 📁 **Multiple Source Support**: Load code from local files or URLs
- 📈 **Real-time Monitoring**: Step-by-step progress visualization
- 🧪 **Automatic Testing**: Intelligent test case generation
- 💾 **Checkpoint System**: Regular saving and restoration

## 📦 Installation

### Requirements
```bash
Python 3.8+
```

### Optional Libraries
```bash
pip install numpy      # For NumPy array support
pip install requests   # For loading code from URLs
```

## 🚀 Usage

### Getting Help
```bash
# Show help in English
python v3.py --help
python v3.py -h
python v3.py help

# Show help in Turkish (Türkçe yardım)
python v3.py --help-tr
python v3.py -htr
python v3.py yardim
```

### Basic Usage
```bash
# From file
python v3.py mycode.py function_name

# From URL
python v3.py https://raw.githubusercontent.com/user/repo/main/code.py function_name

# Interactive mode
python v3.py
```

### Example Usage
```bash
# Optimize a function
python v3.py example.py calculate_sum

# Optimize a method
python v3.py mymodule.py MyClass.process_data
```

## 📖 How It Works

GENESIS v3 operates through a 7-phase process:

### 1️⃣ Loading
- Loads source code from file or URL
- Creates AST (Abstract Syntax Tree)
- Dynamically loads the module

### 2️⃣ Analysis
- Scans for functions and methods
- Analyzes parameter types
- Lists target candidates

### 3️⃣ Target Selection
- User selects the function to optimize
- Displays parameter information

### 4️⃣ Input Generation
- Creates automatic test scenarios
- Type-based intelligent data generation
- 50+ test cases

### 5️⃣ Benchmarking
- Tests the original code
- Collects performance metrics
- Establishes baseline

### 6️⃣ Evolution
- Genetic algorithm loop
- Mutation and selection
- Fitness evaluation
- Preserves best individuals

### 7️⃣ Results
- Saves final code
- Shows statistics
- Lists mutations

## 🧬 Genetic Algorithm

### Mutation Types

1. **Arithmetic Optimizations**
   - `x ** 2` → `x * x` (Multiplication instead of power)
   - `x * 2` → `x + x` (Addition instead of multiplication)
   - `x / const` → `x * (1/const)` (Multiplication instead of division)

2. **Comparison Optimizations**
   - `len(x) == 0` → `not x` (Truthy check instead of length)
   - `x > 0` → `x` (Remove redundant comparison)

3. **Set Operations**
   - `.intersection()` → `&` (Operator instead of method)
   - `.union()` → `|` (Operator instead of method)

### Fitness Function
```python
fitness = (correctness × 10000) + (speedup × 100) - (code_length × 0.01)
```

- **Correctness**: Test pass rate
- **Speedup**: Speed gain compared to original
- **Code Length**: Shorter code is preferred

## 📊 Configuration

Configurable parameters in the `Config` class:

```python
pop_size = 100        # Population size
max_gen = 200         # Maximum generations
elite = 10            # Number of elite individuals
mut_rate = 0.25       # Mutation rate
tourn_size = 5        # Tournament selection size
stag_limit = 30       # Stagnation limit
samples = 50          # Number of test cases
target_spd = 1.5      # Target speedup
```

## 📁 Output Files

After evolution, a folder is created:
```
genesis_function_name_20260101_123456/
├── checkpoint_gen025.py    # Periodic saves
├── checkpoint_gen050.py
├── checkpoint_gen075.py
└── FINAL.py               # Final optimized code
```

Each file contains:
- Generation number
- Fitness score
- Correctness rate
- Speedup metrics
- Applied mutations

## 🎯 Example

### Original Code
```python
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total
```

### After Evolution
```python
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
```

**Results:**
- ✅ 100% correctness
- ⚡ 1.8x speedup
- 📉 15% shorter code

## 🔧 Advanced Usage

### Custom Configuration
```python
from v3 import Config, Evolution

config = Config(
    pop_size=200,      # Larger population
    max_gen=500,       # More generations
    mut_rate=0.35,     # Higher mutation rate
    samples=100        # More test cases
)
```

### Programmatic Use
```python
from v3 import SourceLoader, CodeAnalyzer, Evolution

# Load source
code, path = SourceLoader.load("mycode.py")

# Analyze
analyzer = CodeAnalyzer(code, module)
targets = analyzer.find_targets()

# Evolve
# ... (see source code for full example)
```

## ⚠️ Limitations

- Only works with pure Python code
- Cannot optimize code with external dependencies (unless installed)
- AST mutations are limited to predefined patterns
- Performance gains depend on code structure
- May not always find better solutions

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- More mutation strategies
- Support for more Python constructs
- Multi-objective optimization
- Parallel evaluation
- Better type inference
- Integration with profilers

## 📝 Version History

### v3.0 (Current)
- Complete rewrite with improved architecture
- AST-based mutations
- URL support
- Enhanced error handling
- Better progress visualization

### v2.0
- Added class method support
- Improved fitness function
- Checkpoint system

### v1.0
- Initial release
- Basic function optimization

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**SixFingerDev**

## 🙏 Acknowledgments

- Inspired by genetic programming research
- Built with Python's `ast` module
- Uses tournament selection and elitism

## 📚 References

- [Genetic Algorithms in Python](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Python AST Module](https://docs.python.org/3/library/ast.html)
- [Code Optimization Techniques](https://wiki.python.org/moin/PythonSpeed)

---

## 🇹🇷 Türkçe Özet / Turkish Summary

**Genesis v3**, Python kodunu otomatik olarak optimize eden bir araçtır. Genetik algoritmalar kullanarak fonksiyonlarınızı ve metodlarınızı daha hızlı ve verimli hale getirir.

### Temel Özellikler
- 🎯 Otomatik kod optimizasyonu
- 📊 Performans karşılaştırması
- 🔄 Akıllı mutasyonlar
- 📁 Dosya ve URL desteği
- 📈 Gerçek zamanlı izleme
- 🧪 Otomatik test oluşturma

### Kullanım
```bash
# Yardım göster
python v3.py --help-tr

# Fonksiyon optimize et
python v3.py ornekdosya.py fonksiyon_adi

# İnteraktif mod
python v3.py
```

Daha fazla bilgi için İngilizce README'yi okuyun veya `python v3.py --help-tr` komutunu çalıştırın.

---

**Note:** This is an experimental tool. Always review evolved code before using in production!

