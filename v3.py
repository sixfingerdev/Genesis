#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                         GENESIS v3 - FINAL                                    ║
║                   Universal Code Evolution Engine                             ║
║                                                                               ║
║                          by SixFingerDev                                      ║
║                                                                               ║
║  Features:                                                                    ║
║  • File & URL support                                                         ║
║  • Functions, Classes, Methods                                                ║
║  • Multi-parameter support                                                    ║
║  • Step-by-step progress                                                      ║
║  • Quality over speed                                                         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import ast
import copy
import time
import random
import inspect
import sys
import pickle
import os
from io import StringIO
from typing import Any, List, Tuple, Optional, Callable, Dict, get_type_hints
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import importlib.util
import tempfile

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except:
    HAS_NUMPY = False
    print("⚠️  NumPy not available (optional)")

try:
    import requests
    HAS_REQUESTS = True
except:
    HAS_REQUESTS = False
    print("⚠️  Requests not available (URL support disabled)")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """Terminal colors."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.ENDC = cls.BOLD = ''


def progress_bar(current: int, total: int, width: int = 50, prefix: str = ""):
    """Ilerleme cubugu."""
    percent = current / total if total > 0 else 1
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    percent_str = f"{percent*100:.1f}%"
    print(f"\r{prefix}[{bar}] {current}/{total} ({percent_str})", end="", flush=True)
    if current == total:
        print()


def phase_header(title: str, step: int = 0, total_steps: int = 0):
    """Faz basligi."""
    border = "━" * 75
    if step > 0:
        title = f"[PHASE {step}/{total_steps}] {title}"
    print(f"\n{Colors.CYAN}{border}{Colors.ENDC}")
    print(f"{Colors.BOLD}{title.upper()}{Colors.ENDC}")
    print(f"{Colors.CYAN}{border}{Colors.ENDC}\n")
    time.sleep(0.3)


def step(msg: str, delay: float = 0.2):
    """Adim mesaji."""
    print(f"  {Colors.BLUE}→{Colors.ENDC} {msg}")
    time.sleep(delay)


def success(msg: str, delay: float = 0.15):
    """Basari mesaji."""
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {msg}")
    time.sleep(delay)


def warning(msg: str):
    """Uyari mesaji."""
    print(f"  {Colors.YELLOW}⚠️{Colors.ENDC} {msg}")


def error(msg: str):
    """Hata mesaji."""
    print(f"  {Colors.RED}❌{Colors.ENDC} {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class SourceLoader:
    """Kaynak kod yukleyici (file veya URL)."""
    
    @classmethod
    def load(cls, source: str) -> Tuple[str, Path]:
        """
        Kaynak kodu yukle.
        Returns: (source_code, temp_filepath)
        """
        step("Analyzing source type...")
        
        # URL check
        if source.startswith(('http://', 'https://', 'raw.githubusercontent.com')):
            if not HAS_REQUESTS:
                error("URL support requires 'requests' library")
                error("Install: pip install requests")
                sys.exit(1)
            return cls._load_url(source)
        
        # File check
        path = Path(source)
        if path.exists():
            return cls._load_file(path)
        
        error(f"Source not found: {source}")
        error("Provide a valid file path or URL")
        sys.exit(1)
    
    @classmethod
    def _load_url(cls, url: str) -> Tuple[str, Path]:
        """URL'den yukle."""
        step(f"Downloading from URL...")
        step(f"URL: {url[:60]}..." if len(url) > 60 else f"URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            code = response.text
            
            # Temp file'a kaydet
            temp_file = Path(tempfile.gettempdir()) / f"genesis_source_{int(time.time())}.py"
            temp_file.write_text(code, encoding='utf-8')
            
            success(f"Downloaded {len(code)} characters")
            return code, temp_file
            
        except Exception as e:
            error(f"Failed to download: {e}")
            sys.exit(1)
    
    @classmethod
    def _load_file(cls, path: Path) -> Tuple[str, Path]:
        """Dosyadan yukle."""
        step(f"Reading file: {path}")
        
        try:
            code = path.read_text(encoding='utf-8')
            success(f"Loaded {len(code)} characters")
            return code, path
        except Exception as e:
            error(f"Failed to read file: {e}")
            sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Parameter:
    """Fonksiyon parametresi."""
    name: str
    type: str = 'any'
    default: Any = None
    has_default: bool = False


@dataclass
class Target:
    """Evrimlestirilecek hedef."""
    kind: str           # 'function', 'method', 'class'
    name: str
    obj: Any
    params: List[Parameter]
    parent_class: Any = None
    instance: Any = None
    ast_node: Any = None
    source: str = ""
    
    def __str__(self):
        params_str = ', '.join([f"{p.name}:{p.type}" for p in self.params])
        return f"[{self.kind}] {self.name}({params_str})"


# ═══════════════════════════════════════════════════════════════════════════════
# CODE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class CodeAnalyzer:
    """Kodu analiz et."""
    
    TYPE_PATTERNS = {
        'str': ['query', 'text', 'string', 'name', 'path', 'message', 'word', 'sentence', 'document'],
        'int': ['count', 'size', 'num', 'max', 'min', 'limit', 'k', 'n', 'top_k', 'max_tokens', 'length'],
        'float': ['rate', 'ratio', 'temperature', 'temp', 'alpha', 'beta', 'threshold', 'score', 'weight'],
        'bool': ['is_', 'has_', 'use_', 'enable', 'flag', 'verbose', 'debug', 'show'],
        'list': ['items', 'tokens', 'words', 'elements', 'data', 'values', 'generated', 'documents'],
        'dict': ['config', 'options', 'params', 'kwargs', 'settings', 'mapping'],
        'array': ['tensor', 'vector', 'matrix', 'embedding', 'weights', 'x', 'y', 'arr'],
    }
    
    def __init__(self, source: str, module: Any):
        self.source = source
        self.module = module
        self.tree = ast.parse(source)
    
    def find_targets(self) -> List[Target]:
        """Tum hedefleri bul."""
        step("Scanning for targets...")
        time.sleep(0.3)
        
        targets = []
        
        for name in dir(self.module):
            if name.startswith('_'):
                continue
            
            obj = getattr(self.module, name)
            
            if not callable(obj):
                continue
            if name in ('np', 'numpy', 'pickle', 'random', 'print', 'len', 'range', 'enumerate'):
                continue
            
            # Function
            if inspect.isfunction(obj):
                target = self._analyze_function(name, obj)
                if target:
                    targets.append(target)
                    step(f"Found: {target}", delay=0.1)
            
            # Class
            elif inspect.isclass(obj):
                # Skip built-in classes
                if obj.__module__ == 'builtins':
                    continue
                
                # Analyze methods
                for method_name in dir(obj):
                    if method_name.startswith('_'):
                        continue
                    method = getattr(obj, method_name)
                    if callable(method) and not inspect.isclass(method):
                        target = self._analyze_method(name, obj, method_name, method)
                        if target:
                            targets.append(target)
                            step(f"Found: {target}", delay=0.1)
        
        success(f"Found {len(targets)} targets")
        return targets
    
    def _analyze_function(self, name: str, func: Callable) -> Optional[Target]:
        """Fonksiyonu analiz et."""
        try:
            params = self._extract_params(func)
            ast_node = self._find_ast_node(name, 'function')
            source = ast.unparse(ast_node) if ast_node else ""
            
            return Target(
                kind='function',
                name=name,
                obj=func,
                params=params,
                ast_node=ast_node,
                source=source
            )
        except:
            return None
    
    def _analyze_method(self, class_name: str, cls: type, method_name: str, method: Callable) -> Optional[Target]:
        """Method analiz et."""
        try:
            params = self._extract_params(method)
            # 'self' parametresini cikar
            params = [p for p in params if p.name != 'self']
            
            # Instance olustur
            instance = self._try_create_instance(cls)
            
            ast_node = self._find_method_ast(class_name, method_name)
            source = ast.unparse(ast_node) if ast_node else ""
            
            return Target(
                kind='method',
                name=f"{class_name}.{method_name}",
                obj=method,
                params=params,
                parent_class=cls,
                instance=instance,
                ast_node=ast_node,
                source=source
            )
        except:
            return None
    
    def _extract_params(self, func: Callable) -> List[Parameter]:
        """Parametre bilgilerini cikar."""
        params = []
        
        try:
            sig = inspect.signature(func)
            hints = {}
            try:
                hints = get_type_hints(func)
            except:
                pass
            
            for name, param in sig.parameters.items():
                p = Parameter(name=name)
                
                # Type hint
                if name in hints:
                    p.type = self._type_to_str(hints[name])
                elif param.annotation != inspect.Parameter.empty:
                    p.type = self._type_to_str(param.annotation)
                
                # Default value
                if param.default != inspect.Parameter.empty:
                    p.default = param.default
                    p.has_default = True
                    if p.type == 'any':
                        p.type = self._infer_type_from_value(param.default)
                
                # Guess from name
                if p.type == 'any':
                    p.type = self._guess_type_from_name(name)
                
                params.append(p)
        except:
            pass
        
        return params
    
    def _type_to_str(self, t: Any) -> str:
        """Type annotation'i string'e cevir."""
        s = str(t).lower()
        if 'str' in s:
            return 'str'
        elif 'int' in s:
            return 'int'
        elif 'float' in s:
            return 'float'
        elif 'bool' in s:
            return 'bool'
        elif 'list' in s:
            return 'list'
        elif 'dict' in s:
            return 'dict'
        elif 'ndarray' in s or 'array' in s:
            return 'array'
        return 'any'
    
    def _infer_type_from_value(self, value: Any) -> str:
        """Degerden tip cikar."""
        if isinstance(value, str):
            return 'str'
        elif isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, list):
            return 'list'
        elif isinstance(value, dict):
            return 'dict'
        elif HAS_NUMPY and isinstance(value, np.ndarray):
            return 'array'
        return 'any'
    
    def _guess_type_from_name(self, name: str) -> str:
        """Parametre isminden tip tahmin et."""
        name = name.lower()
        
        for typ, keywords in self.TYPE_PATTERNS.items():
            for kw in keywords:
                if kw in name or name.startswith(kw):
                    return typ
        
        return 'any'
    
    def _try_create_instance(self, cls: type) -> Any:
        """Class instance olustur."""
        try:
            return cls()
        except:
            try:
                instance = object.__new__(cls)
                return instance
            except:
                return None
    
    def _find_ast_node(self, name: str, kind: str) -> Optional[ast.AST]:
        """AST'den node bul."""
        for node in ast.walk(self.tree):
            if kind == 'function' and isinstance(node, ast.FunctionDef):
                if node.name == name:
                    return node
        return None
    
    def _find_method_ast(self, class_name: str, method_name: str) -> Optional[ast.AST]:
        """Method AST'sini bul."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return item
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class InputGenerator:
    """Akilli input generator."""
    
    SAMPLE_STRINGS = [
        "hello world", "Merhaba Dünya", "Türkiye'nin başkenti neresi?",
        "Python programlama", "yapay zeka", "machine learning",
        "test query", "örnek metin", "konya", "istanbul", "ankara",
        "nasılsın", "merhaba", "günaydın", "teşekkürler",
        "How are you?", "What is AI?", "Tell me about Python",
        "", "a", "test", "12345", "special!@#",
    ]
    
    @classmethod
    def generate(cls, params: List[Parameter], count: int = 50) -> List[Tuple]:
        """Input uret."""
        step(f"Generating {count} test inputs...")
        time.sleep(0.2)
        
        if not params:
            success("No parameters (will test execution only)")
            return [() for _ in range(count)]
        
        inputs = []
        for i in range(count):
            args = tuple(cls._generate_value(p) for p in params)
            inputs.append(args)
            
            if (i + 1) % 10 == 0:
                progress_bar(i + 1, count, prefix="  ")
        
        progress_bar(count, count, prefix="  ")
        
        # Show samples
        step("Sample inputs:")
        for i, inp in enumerate(inputs[:3], 1):
            formatted = cls._format_input(inp)
            print(f"     {i}. {formatted}")
            time.sleep(0.1)
        if len(inputs) > 3:
            print(f"     ... ({len(inputs) - 3} more)")
        
        success(f"Generated {len(inputs)} test cases")
        return inputs
    
    @classmethod
    def _generate_value(cls, param: Parameter) -> Any:
        """Tek parametre icin deger uret."""
        # Default varsa bazen kullan
        if param.has_default and random.random() < 0.2:
            return param.default
        
        generators = {
            'str': cls._gen_str,
            'int': cls._gen_int,
            'float': cls._gen_float,
            'bool': cls._gen_bool,
            'list': cls._gen_list,
            'dict': cls._gen_dict,
            'array': cls._gen_array,
            'any': cls._gen_any,
        }
        
        gen_func = generators.get(param.type, cls._gen_any)
        return gen_func()
    
    @classmethod
    def _gen_str(cls) -> str:
        return random.choice(cls.SAMPLE_STRINGS)
    
    @classmethod
    def _gen_int(cls) -> int:
        return random.choice([0, 1, 5, 10, 20, 50, 100, random.randint(1, 100)])
    
    @classmethod
    def _gen_float(cls) -> float:
        return random.choice([0.0, 0.5, 0.7, 1.0, random.uniform(0, 1)])
    
    @classmethod
    def _gen_bool(cls) -> bool:
        return random.choice([True, False])
    
    @classmethod
    def _gen_list(cls) -> List:
        size = random.randint(0, 15)
        return [random.randint(0, 100) for _ in range(size)]
    
    @classmethod
    def _gen_dict(cls) -> Dict:
        size = random.randint(0, 10)
        return {f"key_{i}": random.randint(0, 100) for i in range(size)}
    
    @classmethod
    def _gen_array(cls):
        if not HAS_NUMPY:
            return cls._gen_list()
        size = random.choice([10, 50, 100])
        return np.random.randn(size).astype(np.float32)
    
    @classmethod
    def _gen_any(cls) -> Any:
        t = random.choice(['str', 'int', 'list'])
        return {'str': cls._gen_str, 'int': cls._gen_int, 'list': cls._gen_list}[t]()
    
    @classmethod
    def _format_input(cls, inp: Tuple) -> str:
        """Input'u okunabilir formatta goster."""
        if not inp:
            return "()"
        
        parts = []
        for item in inp:
            if isinstance(item, str):
                parts.append(f'"{item[:30]}..."' if len(item) > 30 else f'"{item}"')
            elif isinstance(item, (list, tuple)) and len(item) > 5:
                parts.append(f"[...{len(item)} items]")
            elif isinstance(item, dict) and len(item) > 3:
                parts.append(f"{{...{len(item)} keys}}")
            elif HAS_NUMPY and isinstance(item, np.ndarray):
                parts.append(f"array({item.shape})")
            else:
                parts.append(str(item))
        
        return f"({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class Executor:
    """Guvenli kod calistirici."""
    
    @classmethod
    def execute(cls, target: Target, args: Tuple, timeout: float = 2.0) -> Tuple[Any, float, Optional[str]]:
        """Hedefi calistir."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = StringIO(), StringIO()
        
        result, elapsed, error = None, 0.0, None
        
        try:
            start = time.perf_counter()
            
            if target.kind == 'function':
                result = target.obj(*args)
            elif target.kind == 'method' and target.instance:
                bound_method = getattr(target.instance, target.name.split('.')[-1])
                result = bound_method(*args)
            else:
                error = "Cannot execute"
            
            elapsed = (time.perf_counter() - start) * 1000
        except Exception as e:
            error = str(e)
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
        
        return result, elapsed, error


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Comparator:
    """Universal karsilastirma."""
    
    @classmethod
    def compare(cls, result: Any, expected: Any, rtol: float = 1e-5) -> bool:
        try:
            if result is None and expected is None:
                return True
            if result is None or expected is None:
                return False
            
            # Numbers
            if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
                return abs(float(result) - float(expected)) < rtol * (abs(float(expected)) + 1)
            
            # Numpy
            if HAS_NUMPY and isinstance(expected, np.ndarray):
                if not isinstance(result, np.ndarray):
                    return False
                return np.allclose(result, expected, rtol=rtol, equal_nan=True)
            
            # Strings
            if isinstance(expected, str):
                return str(result) == expected
            
            # Lists
            if isinstance(expected, (list, tuple)):
                if not isinstance(result, (list, tuple)) or len(result) != len(expected):
                    return False
                return all(cls.compare(r, e, rtol) for r, e in zip(result, expected))
            
            # Dicts
            if isinstance(expected, dict):
                if not isinstance(result, dict) or set(result.keys()) != set(expected.keys()):
                    return False
                return all(cls.compare(result[k], expected[k], rtol) for k in expected)
            
            # Objects
            if hasattr(expected, '__dict__') and hasattr(result, '__dict__'):
                return True
            
            return result == expected
        except:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARKER
# ═══════════════════════════════════════════════════════════════════════════════

class Benchmarker:
    """Orijinal kodu benchmark et."""
    
    @classmethod
    def benchmark(cls, target: Target, inputs: List[Tuple]) -> Tuple[List[Any], List[float]]:
        """Benchmark yap."""
        step("Running original function on test cases...")
        time.sleep(0.2)
        
        outputs = []
        times = []
        
        for i, args in enumerate(inputs):
            # Deep copy
            args_copy = []
            for a in args:
                if hasattr(a, 'copy'):
                    args_copy.append(a.copy())
                elif isinstance(a, (list, dict)):
                    args_copy.append(copy.deepcopy(a))
                else:
                    args_copy.append(a)
            args_copy = tuple(args_copy)
            
            result, elapsed, error = Executor.execute(target, args_copy)
            outputs.append(result)
            times.append(elapsed if error is None else float('inf'))
            
            if (i + 1) % 5 == 0:
                progress_bar(i + 1, len(inputs), prefix="  ")
        
        progress_bar(len(inputs), len(inputs), prefix="  ")
        
        # Statistics
        valid = [t for t in times if t < float('inf')]
        success_count = len(valid)
        success_rate = success_count / len(times)
        
        if valid:
            avg_time = sum(valid) / len(valid)
            min_time = min(valid)
            max_time = max(valid)
            
            step("Benchmark results:")
            print(f"     Average: {avg_time:.4f} ms")
            print(f"     Min: {min_time:.4f} ms")
            print(f"     Max: {max_time:.4f} ms")
            print(f"     Success: {success_count}/{len(times)} ({success_rate:.0%})")
            time.sleep(0.3)
            
            if success_rate < 0.5:
                warning(f"Low success rate ({success_rate:.0%})")
                warning("Evolution may be difficult")
            
            success("Benchmarking complete")
        else:
            error("All tests failed!")
            error("Cannot proceed with evolution")
            return None, None
        
        return outputs, times


# ═══════════════════════════════════════════════════════════════════════════════
# AST MUTATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Mutator(ast.NodeTransformer):
    """AST mutasyonlari."""
    
    def __init__(self, rate: float = 0.15):
        super().__init__()
        self.rate = rate
        self.mutations = []
    
    def mutate(self) -> bool:
        return random.random() < self.rate
    
    def log(self, msg: str):
        self.mutations.append(msg)
    
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not self.mutate():
            return node
        
        # x ** 2 → x * x
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                self.log("x**2 → x*x")
                return ast.BinOp(
                    left=copy.deepcopy(node.left),
                    op=ast.Mult(),
                    right=copy.deepcopy(node.left)
                )
        
        # x * 2 → x + x
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 2:
                self.log("x*2 → x+x")
                return ast.BinOp(
                    left=copy.deepcopy(node.left),
                    op=ast.Add(),
                    right=copy.deepcopy(node.left)
                )
        
        # x / const → x * (1/const)
        if isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant):
                if isinstance(node.right.value, (int, float)) and node.right.value != 0:
                    if random.random() < 0.3:
                        inv = 1.0 / node.right.value
                        self.log(f"/{node.right.value} → *{inv:.6f}")
                        return ast.BinOp(
                            left=node.left,
                            op=ast.Mult(),
                            right=ast.Constant(value=inv)
                        )
        
        return node
    
    def visit_Compare(self, node):
        self.generic_visit(node)
        if not self.mutate():
            return node
        
        if len(node.ops) != 1:
            return node
        
        # len(x) == 0 → not x
        if isinstance(node.ops[0], ast.Eq):
            if isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 0:
                if isinstance(node.left, ast.Call):
                    if isinstance(node.left.func, ast.Name) and node.left.func.id == 'len':
                        if node.left.args:
                            self.log("len(x)==0 → not x")
                            return ast.UnaryOp(op=ast.Not(), operand=node.left.args[0])
        
        # x > 0 → x (for union check)
        if isinstance(node.ops[0], ast.Gt):
            if isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 0:
                if random.random() < 0.2:
                    self.log("x>0 → x (truthy)")
                    return node.left
        
        return node
    
    def visit_Call(self, node):
        self.generic_visit(node)
        if not self.mutate():
            return node
        
        if isinstance(node.func, ast.Attribute):
            # .intersection() → & operator
            if node.func.attr == 'intersection' and node.args:
                if random.random() < 0.3:
                    self.log(".intersection() → &")
                    return ast.BinOp(
                        left=node.func.value,
                        op=ast.BitAnd(),
                        right=node.args[0]
                    )
            
            # .union() → | operator
            if node.func.attr == 'union' and node.args:
                if random.random() < 0.3:
                    self.log(".union() → |")
                    return ast.BinOp(
                        left=node.func.value,
                        op=ast.BitOr(),
                        right=node.args[0]
                    )
            
            # .strip().strip() → .strip()
            if node.func.attr == 'strip':
                if isinstance(node.func.value, ast.Call):
                    if isinstance(node.func.value.func, ast.Attribute):
                        if node.func.value.func.attr == 'strip':
                            self.log("strip().strip() → strip()")
                            return node.func.value
        
        return node


def mutate_ast(tree, rate=0.15):
    m = Mutator(rate)
    new = m.visit(copy.deepcopy(tree))
    ast.fix_missing_locations(new)
    return new, m.mutations


# ═══════════════════════════════════════════════════════════════════════════════
# GENOME
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Genome:
    """Genom - evrimleşen kod birimi."""
    ast_tree: ast.AST
    source: str = ""
    gen: int = 0
    mutations: List[str] = field(default_factory=list)
    fitness: float = 0.0
    correctness: float = 0.0
    time_ms: float = float('inf')
    speedup: float = 0.0
    
    def __post_init__(self):
        if not self.source:
            try:
                self.source = ast.unparse(self.ast_tree)
            except:
                pass
    
    def compile_function(self, name: str, globs: Dict = None) -> Optional[Callable]:
        """Fonksiyonu compile et."""
        try:
            code = compile(ast.Module(body=[self.ast_tree], type_ignores=[]), '<evolved>', 'exec')
            ns = {'__builtins__': __builtins__}
            if HAS_NUMPY:
                ns['np'] = ns['numpy'] = np
            ns['random'] = random
            ns['pickle'] = pickle
            if globs:
                ns.update(globs)
            exec(code, ns)
            return ns.get(name.split('.')[-1])
        except:
            return None
    
    def clone(self):
        return Genome(
            ast_tree=copy.deepcopy(self.ast_tree),
            gen=self.gen,
            mutations=list(self.mutations)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Evrim konfigurasyonu."""
    pop_size: int = 100
    max_gen: int = 200
    elite: int = 10
    mut_rate: float = 0.25
    tourn_size: int = 5
    stag_limit: int = 30
    samples: int = 50
    target_spd: float = 1.5
    
    # Display
    progress_every: int = 1
    checkpoint_every: int = 25


# ═══════════════════════════════════════════════════════════════════════════════
# ARENA
# ═══════════════════════════════════════════════════════════════════════════════

class Arena:
    """Evrim arenasi - genomlari test et."""
    
    def __init__(self, target: Target, inputs: List[Tuple], outputs: List,
                 times: List[float], globs: Dict = None):
        self.target = target
        self.inputs = inputs
        self.outputs = outputs
        self.times = times
        valid = [t for t in times if t < float('inf')]
        self.orig_avg = sum(valid) / len(valid) if valid else 1.0
        self.globs = globs or {}
    
    def evaluate(self, g: Genome) -> Genome:
        """Genomu degerlendir."""
        # Compile
        func = g.compile_function(self.target.name, self.globs)
        if not func:
            g.fitness, g.correctness = -1000, 0.0
            return g
        
        ok, total_t, valid = 0, 0.0, 0
        
        for args, exp in zip(self.inputs, self.outputs):
            # Deep copy args
            args_copy = []
            for a in args:
                if hasattr(a, 'copy'):
                    args_copy.append(a.copy())
                elif isinstance(a, (list, dict)):
                    args_copy.append(copy.deepcopy(a))
                else:
                    args_copy.append(a)
            args_copy = tuple(args_copy)
            
            try:
                # Execute
                if self.target.kind == 'method' and self.target.instance:
                    bound = lambda *a, f=func, inst=self.target.instance: f(inst, *a)
                    temp_target = Target(kind='function', name='', obj=bound, params=[])
                    res, t, err = Executor.execute(temp_target, args_copy)
                else:
                    temp_target = Target(kind='function', name='', obj=func, params=[])
                    res, t, err = Executor.execute(temp_target, args_copy)
                
                if err is None and Comparator.compare(res, exp):
                    ok += 1
                    total_t += t
                    valid += 1
            except:
                pass
        
        if valid > 0:
            g.correctness = ok / len(self.inputs)
            g.time_ms = total_t / valid
            g.speedup = self.orig_avg / g.time_ms if g.time_ms > 0 else 0
        else:
            g.correctness, g.time_ms, g.speedup = 0.0, float('inf'), 0.0
        
        g.fitness = (
            g.correctness * 10000 +
            g.speedup * 100 -
            len(g.source) * 0.01
        )
        
        if g.correctness < 1.0:
            g.fitness *= g.correctness
        
        return g


# ═══════════════════════════════════════════════════════════════════════════════
# EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

class Evolution:
    """Evrim motoru."""
    
    def __init__(self, cfg: Config, arena: Arena, orig: Genome, out_dir: Path):
        self.cfg = cfg
        self.arena = arena
        self.orig = orig
        self.out = out_dir
        self.gen = 0
        self.pop = []
        self.best = None
        self.stag = 0
    
    def init(self):
        """Ilk populasyonu olustur."""
        step(f"Initializing population ({self.cfg.pop_size} individuals)...")
        time.sleep(0.3)
        
        self.pop = [self.orig.clone()]
        
        for i in range(self.cfg.pop_size - 1):
            self.pop.append(self._mutate(self.orig, 0))
            if (i + 1) % 20 == 0:
                progress_bar(i + 1, self.cfg.pop_size - 1, prefix="  ")
        
        progress_bar(self.cfg.pop_size - 1, self.cfg.pop_size - 1, prefix="  ")
        success(f"Population ready ({len(self.pop)} individuals)")
    
    def _mutate(self, parent: Genome, gen: int) -> Genome:
        """Mutant olustur."""
        rate = self.cfg.mut_rate * random.uniform(0.8, 1.5)
        tree, muts = mutate_ast(parent.ast_tree, rate)
        return Genome(ast_tree=tree, gen=gen, mutations=muts)
    
    def _select(self) -> Genome:
        """Turnuva secimi."""
        cands = random.sample(self.pop, min(self.cfg.tourn_size, len(self.pop)))
        return max(cands, key=lambda g: g.fitness)
    
    def step(self) -> Dict:
        """Bir nesil ilerlet."""
        self.gen += 1
        
        # Evaluate all
        for i, g in enumerate(self.pop):
            self.arena.evaluate(g)
            progress_bar(i + 1, len(self.pop), prefix=f"  Gen {self.gen} evaluating: ")
        
        # Sort by fitness
        self.pop.sort(key=lambda g: g.fitness, reverse=True)
        
        best = self.pop[0]
        
        # Update best ever
        if self.best is None or best.fitness > self.best.fitness:
            self.best = best.clone()
            self.best.fitness = best.fitness
            self.best.correctness = best.correctness
            self.best.speedup = best.speedup
            self.best.time_ms = best.time_ms
            self.stag = 0
        else:
            self.stag += 1
        
        # New generation
        new = [self.pop[i].clone() for i in range(min(self.cfg.elite, len(self.pop)))]
        while len(new) < self.cfg.pop_size:
            new.append(self._mutate(self._select(), self.gen))
        self.pop = new
        
        return {
            'gen': self.gen,
            'fitness': best.fitness,
            'correctness': best.correctness,
            'speedup': best.speedup,
            'time_ms': best.time_ms
        }
    
    def stop(self) -> Tuple[bool, str]:
        """Durma kontrolu."""
        if self.gen >= self.cfg.max_gen:
            return True, "Maximum generations reached"
        if self.stag >= self.cfg.stag_limit:
            return True, f"Stagnation ({self.stag} generations)"
        if self.best and self.best.correctness >= 1.0:
            if self.best.speedup >= self.cfg.target_spd:
                return True, f"Target achieved! ({self.cfg.target_spd}x speedup)"
        return False, ""
    
    def save_checkpoint(self, g: Genome) -> Path:
        """Checkpoint kaydet."""
        filename = f"checkpoint_gen{self.gen:03d}.py"
        filepath = self.out / filename
        content = f"# Gen {self.gen} | Fitness: {g.fitness:.2f} | Speedup: {g.speedup:.2f}x\n\n{g.source}"
        filepath.write_text(content)
        return filepath
    
    def save_final(self, g: Genome) -> Path:
        """Final kaydet."""
        filepath = self.out / "FINAL.py"
        
        header = f"""# ═════════════════════════════════════════════════════════════════════════════
# GENESIS v3 - Evolved Code
# ═════════════════════════════════════════════════════════════════════════════
#
# Generation: {self.gen}
# Final Fitness: {g.fitness:.2f}
# Correctness: {g.correctness:.2%}
# Speedup: {g.speedup:.2f}x
# Execution Time: {g.time_ms:.4f} ms
#
# Mutations Applied:
"""
        
        for i, mut in enumerate(g.mutations, 1):
            header += f"#   {i}. {mut}\n"
        
        header += "#\n# ═════════════════════════════════════════════════════════════════════════════\n\n"
        
        content = header + g.source
        filepath.write_text(content)
        return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Ana program."""
    
    # Banner
    print(f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                         GENESIS v3 - FINAL                                    ║
║                   Universal Code Evolution Engine                             ║
║                                                                               ║
║                          by SixFingerDev                                      ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
    """)
    
    time.sleep(0.5)
    
    # Total phases
    TOTAL_PHASES = 7
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: LOADING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    phase_header("Loading", 1, TOTAL_PHASES)
    
    # Get source
    if len(sys.argv) > 1:
        source_input = sys.argv[1]
    else:
        source_input = input(f"  {Colors.BOLD}Enter file path or URL:{Colors.ENDC} ").strip()
    
    if not source_input:
        error("No source provided")
        sys.exit(1)
    
    print()
    source_code, filepath = SourceLoader.load(source_input)
    
    step("Parsing AST...")
    time.sleep(0.2)
    try:
        tree = ast.parse(source_code)
        success("AST parsed successfully")
    except Exception as e:
        error(f"Failed to parse: {e}")
        sys.exit(1)
    
    step("Loading module...")
    time.sleep(0.2)
    try:
        spec = importlib.util.spec_from_file_location("target_module", filepath)
        module = importlib.util.module_from_spec(spec)
        
        # Setup globals
        if HAS_NUMPY:
            module.__dict__['np'] = module.__dict__['numpy'] = np
        module.__dict__['random'] = random
        module.__dict__['pickle'] = pickle
        
        spec.loader.exec_module(module)
        success("Module loaded")
    except Exception as e:
        warning(f"Module execution warning: {e}")
        warning("Continuing with partial load...")
    
    success("Loading phase complete")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 2: ANALYSIS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    phase_header("Analysis", 2, TOTAL_PHASES)
    
    analyzer = CodeAnalyzer(source_code, module)
    targets = analyzer.find_targets()
    
    if not targets:
        error("No targets found for evolution")
        error("Make sure the file contains functions or methods")
        sys.exit(1)
    
    success(f"Analysis phase complete")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: TARGET SELECTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    phase_header("Target Selection", 3, TOTAL_PHASES)
    
    step("Available targets:")
    for i, t in enumerate(targets, 1):
        print(f"     {i}. {t}")
        time.sleep(0.1)
    
    print()
    
    # Select target
    if len(sys.argv) > 2:
        target_name = sys.argv[2]
        selected = next((t for t in targets if t.name == target_name or t.name.endswith(target_name)), None)
        if not selected:
            error(f"Target not found: {target_name}")
            sys.exit(1)
    else:
        if len(targets) == 1:
            selected = targets[0]
            step(f"Auto-selected: {selected}")
        else:
            choice_input = input(f"  {Colors.BOLD}Select target (1-{len(targets)}):{Colors.ENDC} ").strip()
            try:
                idx = int(choice_input) - 1
                selected = targets[idx]
            except:
                error("Invalid selection")
                sys.exit(1)
    
    print()
    step(f"Selected: {Colors.BOLD}{selected.name}{Colors.ENDC}")
    step(f"Kind: {selected.kind}")
    step(f"Parameters: {len(selected.params)}")
    for p in selected.params:
        print(f"     • {p.name}: {p.type}" + (f" = {p.default}" if p.has_default else ""))
        time.sleep(0.1)
    
    success("Target selection complete")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: INPUT GENERATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    phase_header("Input Generation", 4, TOTAL_PHASES)
    
    config = Config()
    inputs = InputGenerator.generate(selected.params, config.samples)
    
    success("Input generation complete")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 5: BENCHMARKING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    phase_header("Benchmarking", 5, TOTAL_PHASES)
    
    outputs, times = Benchmarker.benchmark(selected, inputs)
    
    if outputs is None or times is None:
        error("Benchmarking failed")
        sys.exit(1)
    
    valid_times = [t for t in times if t < float('inf')]
    orig_avg = sum(valid_times) / len(valid_times)
    
    success("Benchmarking phase complete")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 6: EVOLUTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    phase_header("Evolution", 6, TOTAL_PHASES)
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = selected.name.replace('.', '_')
    out_dir = Path.cwd() / f"genesis_{safe_name}_{timestamp}"
    out_dir.mkdir(exist_ok=True)
    step(f"Output directory: {out_dir}")
    
    # Create original genome
    if not selected.ast_node:
        error("AST node not found")
        sys.exit(1)
    
    orig = Genome(ast_tree=selected.ast_node, source=selected.source)
    
    # Setup arena
    globs = {'random': random, 'pickle': pickle}
    if HAS_NUMPY:
        globs['np'] = globs['numpy'] = np
    
    arena = Arena(selected, inputs, outputs, times, globs)
    evo = Evolution(config, arena, orig, out_dir)
    
    # Initialize
    evo.init()
    
    print()
    step(f"Starting evolution (max {config.max_gen} generations)...")
    step("Press Ctrl+C to stop early")
    print()
    time.sleep(0.5)
    
    # Evolution loop
    try:
        while True:
            stats = evo.step()
            
            # Display
            print(f"\n  {Colors.BOLD}Generation {stats['gen']}/{config.max_gen}{Colors.ENDC}")
            print(f"     Fitness:     {stats['fitness']:.2f}")
            print(f"     Correctness: {stats['correctness']:.1%}")
            print(f"     Speedup:     {stats['speedup']:.2f}x")
            print(f"     Time:        {stats['time_ms']:.4f} ms")
            
            if stats['correctness'] >= 1.0:
                print(f"     {Colors.GREEN}✓ Perfect correctness!{Colors.ENDC}")
            
            # Checkpoint
            if stats['gen'] % config.checkpoint_every == 0:
                cp = evo.save_checkpoint(evo.best)
                step(f"Checkpoint saved: {cp.name}")
            
            time.sleep(0.3)
            
            # Check stop
            should_stop, reason = evo.stop()
            if should_stop:
                print(f"\n  {Colors.YELLOW}⏹️  {reason}{Colors.ENDC}\n")
                break
            
    except KeyboardInterrupt:
        print(f"\n\n  {Colors.YELLOW}⚠️  Evolution stopped by user{Colors.ENDC}\n")
    
    success("Evolution phase complete")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 7: RESULTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    phase_header("Results", 7, TOTAL_PHASES)
    
    if not evo.best:
        error("No valid genome found")
        sys.exit(1)
    
    final_path = evo.save_final(evo.best)
    
    # Results box
    border = "═" * 75
    print(f"{Colors.GREEN}{border}{Colors.ENDC}")
    print(f"{Colors.BOLD}  🏆 EVOLUTION COMPLETE{Colors.ENDC}")
    print(f"{Colors.GREEN}{border}{Colors.ENDC}")
    print(f"  Target:         {selected.name}")
    print(f"  Generations:    {evo.gen}")
    print(f"  Final Fitness:  {evo.best.fitness:.2f}")
    print(f"  Correctness:    {evo.best.correctness:.2%}")
    print(f"  Speedup:        {evo.best.speedup:.2f}x")
    print(f"  Time:           {evo.best.time_ms:.4f} ms  (Original: {orig_avg:.4f} ms)")
    print(f"{Colors.GREEN}{border}{Colors.ENDC}")
    
    if evo.best.mutations:
        print(f"\n  {Colors.BOLD}Mutations Applied:{Colors.ENDC}")
        for i, mut in enumerate(evo.best.mutations, 1):
            print(f"     {i}. {mut}")
    
    print(f"\n  {Colors.BOLD}📁 Saved:{Colors.ENDC} {final_path}")
    
    # Show evolved code
    print(f"\n  {Colors.BOLD}Evolved Code:{Colors.ENDC}")
    print(f"  {Colors.CYAN}{'─' * 73}{Colors.ENDC}")
    for line in evo.best.source.split('\n'):
        print(f"  │ {line}")
    print(f"  {Colors.CYAN}{'─' * 73}{Colors.ENDC}")
    
    print(f"\n{Colors.GREEN}✨ GENESIS v3 - Evolution Complete ✨{Colors.ENDC}\n")


if __name__ == "__main__":
    main()