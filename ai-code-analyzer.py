"""
AI Code Quality Analyzer
A production-ready tool for analyzing AI-generated code using prompt engineering techniques.
Author: Mark Cox
"""

import os
import ast
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a code quality issue"""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'syntax', 'style', 'logic', 'security', 'performance'
    line: int
    description: str
    suggestion: str


@dataclass
class AnalysisResult:
    """Complete analysis result for a code file"""
    file_path: str
    language: str
    overall_score: float
    ai_confidence: float
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    timestamp: str
    

class PromptTemplates:
    """Centralized prompt templates for code analysis"""
    
    CODE_REVIEW = """
    Analyze the following {language} code for quality issues. Focus on:
    1. Syntax and semantic correctness
    2. Code style and best practices
    3. Potential bugs or logic errors
    4. Security vulnerabilities
    5. Performance concerns
    
    Code to analyze:
    ```{language}
    {code}
    ```
    
    Provide analysis in JSON format with this structure:
    {{
        "overall_quality": "score 0-100",
        "ai_indicators": "confidence that this is AI-generated (0-100)",
        "issues": [
            {{
                "severity": "critical|warning|info",
                "category": "syntax|style|logic|security|performance",
                "line": line_number,
                "description": "issue description",
                "suggestion": "how to fix"
            }}
        ],
        "strengths": ["list of good practices found"],
        "summary": "brief overall assessment"
    }}
    """
    
    COMPLEXITY_ANALYSIS = """
    Analyze the complexity of this {language} code:
    ```{language}
    {code}
    ```
    
    Calculate:
    1. Cyclomatic complexity
    2. Cognitive complexity
    3. Lines of code metrics
    4. Function/method count
    5. Average function length
    """
    
    SECURITY_SCAN = """
    Perform a security analysis on this {language} code:
    ```{language}
    {code}
    ```
    
    Check for:
    1. SQL injection vulnerabilities
    2. Input validation issues
    3. Hardcoded credentials
    4. Insecure cryptography
    5. Path traversal risks
    6. XSS vulnerabilities (if applicable)
    """


class CodeMetricsCalculator:
    """Calculate various code metrics"""
    
    @staticmethod
    def calculate_python_metrics(code: str) -> Dict[str, Any]:
        """Calculate metrics for Python code"""
        metrics = {
            "lines_of_code": len(code.splitlines()),
            "blank_lines": len([line for line in code.splitlines() if not line.strip()]),
            "comment_lines": 0,
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "complexity": 0
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics["functions"] += 1
                    metrics["complexity"] += CodeMetricsCalculator._calculate_complexity(node)
                elif isinstance(node, ast.ClassDef):
                    metrics["classes"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics["imports"] += 1
                    
            # Count comment lines
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith('#') and not stripped.startswith('#!'):
                    metrics["comment_lines"] += 1
                    
        except SyntaxError:
            logger.warning("Syntax error in Python code, metrics may be incomplete")
            
        return metrics
    
    @staticmethod
    def _calculate_complexity(node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a Python AST node"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += len(child.items)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    @staticmethod
    def calculate_cpp_metrics(code: str) -> Dict[str, Any]:
        """Calculate metrics for C++ code"""
        metrics = {
            "lines_of_code": len(code.splitlines()),
            "blank_lines": len([line for line in code.splitlines() if not line.strip()]),
            "comment_lines": 0,
            "functions": 0,
            "classes": 0,
            "includes": 0,
            "complexity": 0
        }
        
        # Simple regex-based analysis for C++
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith('//') or ('/*' in stripped and '*/' in stripped):
                metrics["comment_lines"] += 1
            elif stripped.startswith('#include'):
                metrics["includes"] += 1
                
        # Count functions (simplified)
        function_pattern = r'\b(?:void|int|double|float|char|bool|auto)\s+\w+\s*\([^)]*\)\s*{'
        metrics["functions"] = len(re.findall(function_pattern, code))
        
        # Count classes/structs
        class_pattern = r'\b(?:class|struct)\s+\w+\s*[:{]'
        metrics["classes"] = len(re.findall(class_pattern, code))
        
        return metrics


class AIPatternDetector:
    """Detect patterns common in AI-generated code"""
    
    AI_PATTERNS = {
        "generic_names": [
            r'\b(?:var|val|temp|data|result|output|input)\d*\b',
            r'\b(?:func|function|method|process)\d*\b',
            r'\bexample_\w+\b',
            r'\bsample_\w+\b'
        ],
        "overly_verbose_comments": [
            r'#\s*This (?:function|method|class) (?:is used to|will)',
            r'#\s*Here we (?:are|will)',
            r'#\s*Now we (?:need to|will|can)',
            r'#\s*First,? we',
            r'#\s*Initialize the'
        ],
        "placeholder_text": [
            r'TODO:?\s*(?:implement|add|complete)',
            r'FIXME:?\s*',
            r'<.*?>',  # HTML-like placeholders
            r'\[.*?\]',  # Bracket placeholders
            r'Your .* here'
        ],
        "common_ai_structures": [
            r'if __name__ == ["\']__main__["\']:',  # Very common in AI Python
            r'def main\(\):',  # Generic main function
            r'except Exception as e:',  # Generic exception handling
            r'print\(["\']Error:?["\'], e\)'  # Generic error printing
        ]
    }
    
    @staticmethod
    def calculate_ai_confidence(code: str) -> Tuple[float, List[str]]:
        """Calculate confidence that code is AI-generated"""
        indicators = []
        pattern_matches = 0
        total_patterns = 0
        
        for category, patterns in AIPatternDetector.AI_PATTERNS.items():
            for pattern in patterns:
                total_patterns += 1
                if re.search(pattern, code, re.IGNORECASE):
                    pattern_matches += 1
                    indicators.append(f"{category}: {pattern}")
        
        # Check for perfect indentation (common in AI code)
        lines = code.splitlines()
        if lines:
            indent_consistency = AIPatternDetector._check_indent_consistency(lines)
            if indent_consistency > 0.95:
                indicators.append("Perfect indentation consistency")
                pattern_matches += 2
        
        # Check for lack of personal style
        personal_style_score = AIPatternDetector._check_personal_style(code)
        if personal_style_score < 0.2:
            indicators.append("Lack of personal coding style")
            pattern_matches += 2
        
        confidence = min((pattern_matches / max(total_patterns, 1)) * 100, 95)
        return confidence, indicators
    
    @staticmethod
    def _check_indent_consistency(lines: List[str]) -> float:
        """Check how consistent indentation is"""
        indents = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indents.append(indent)
        
        if not indents:
            return 1.0
            
        # Check if all indents are multiples of 4 (or 2)
        base_indent = min(indents) if indents else 4
        consistent = sum(1 for i in indents if i % base_indent == 0)
        
        return consistent / len(indents)
    
    @staticmethod
    def _check_personal_style(code: str) -> float:
        """Check for personal coding style indicators"""
        style_indicators = [
            r'#\s*[A-Z]{2,}:',  # Personal comment tags
            r'#\s*@\w+',  # Personal annotations
            r'"""\s*\w+:',  # Personal docstring style
            r'#\s*-{3,}',  # Section separators
            r'(?:fuck|shit|damn|hack)',  # Informal language (censored in production)
        ]
        
        matches = 0
        for pattern in style_indicators:
            if re.search(pattern, code, re.IGNORECASE):
                matches += 1
                
        return matches / len(style_indicators)


class MockAIAnalyzer:
    """Mock AI analyzer for demonstration (replace with actual API calls)"""
    
    @staticmethod
    def analyze_code(code: str, language: str, prompt_template: str) -> Dict[str, Any]:
        """
        Mock analysis - in production, this would call OpenAI/Claude API
        For demonstration, returns realistic mock data
        """
        # In production, you would:
        # 1. Format the prompt with code and language
        # 2. Call OpenAI/Claude API
        # 3. Parse the JSON response
        
        # Mock response based on code length and patterns
        lines = code.splitlines()
        has_issues = len(lines) > 20 or 'TODO' in code or 'FIXME' in code
        
        mock_response = {
            "overall_quality": 75 if has_issues else 85,
            "ai_indicators": 60 if 'def main():' in code else 30,
            "issues": [],
            "strengths": [
                "Clear function naming",
                "Proper error handling",
                "Good code structure"
            ],
            "summary": "Code shows good structure with minor improvement opportunities."
        }
        
        if has_issues:
            mock_response["issues"].extend([
                {
                    "severity": "warning",
                    "category": "style",
                    "line": 10,
                    "description": "Function complexity is high",
                    "suggestion": "Consider breaking into smaller functions"
                },
                {
                    "severity": "info",
                    "category": "performance",
                    "line": 15,
                    "description": "Inefficient loop structure",
                    "suggestion": "Use list comprehension for better performance"
                }
            ])
            
        if 'TODO' in code or 'FIXME' in code:
            mock_response["issues"].append({
                "severity": "warning",
                "category": "logic",
                "line": 1,
                "description": "Incomplete implementation found",
                "suggestion": "Complete TODO/FIXME items before production"
            })
            
        return mock_response


class AICodeAnalyzer:
    """Main analyzer class that orchestrates the analysis"""
    
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.h': 'cpp',
        '.hpp': 'cpp'
    }
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        self.metrics_calculator = CodeMetricsCalculator()
        self.ai_detector = AIPatternDetector()
        self.ai_analyzer = MockAIAnalyzer()
        
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze a single code file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported file type: {extension}")
            
        language = self.SUPPORTED_LANGUAGES[extension]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        return self.analyze_code(code, language, file_path)
    
    def analyze_code(self, code: str, language: str, file_path: str = "inline") -> AnalysisResult:
        """Analyze code content"""
        logger.info(f"Analyzing {language} code from {file_path}")
        
        # Calculate metrics
        if language == 'python':
            metrics = self.metrics_calculator.calculate_python_metrics(code)
        else:
            metrics = self.metrics_calculator.calculate_cpp_metrics(code)
            
        # Detect AI patterns
        ai_confidence, ai_indicators = self.ai_detector.calculate_ai_confidence(code)
        metrics['ai_indicators'] = ai_indicators
        
        # Perform AI analysis if enabled
        issues = []
        overall_score = 85.0  # Default
        
        if self.use_ai:
            try:
                ai_result = self.ai_analyzer.analyze_code(
                    code, 
                    language,
                    PromptTemplates.CODE_REVIEW
                )
                overall_score = ai_result.get('overall_quality', 85)
                
                # Convert AI issues to CodeIssue objects
                for issue in ai_result.get('issues', []):
                    issues.append(CodeIssue(**issue))
                    
                metrics['ai_analysis'] = {
                    'strengths': ai_result.get('strengths', []),
                    'summary': ai_result.get('summary', '')
                }
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                
        # Perform static analysis
        static_issues = self._perform_static_analysis(code, language)
        issues.extend(static_issues)
        
        # Calculate final score
        if issues:
            severity_weights = {'critical': 15, 'warning': 5, 'info': 1}
            total_penalty = sum(severity_weights.get(issue.severity, 0) for issue in issues)
            overall_score = max(0, overall_score - total_penalty)
        
        return AnalysisResult(
            file_path=file_path,
            language=language,
            overall_score=overall_score,
            ai_confidence=ai_confidence,
            issues=issues,
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def _perform_static_analysis(self, code: str, language: str) -> List[CodeIssue]:
        """Perform basic static analysis"""
        issues = []
        
        # Check for common issues
        lines = code.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 100:
                issues.append(CodeIssue(
                    severity='info',
                    category='style',
                    line=i,
                    description=f'Line too long ({len(line)} characters)',
                    suggestion='Break line to improve readability (max 100 chars)'
                ))
                
            # Trailing whitespace
            if line != line.rstrip():
                issues.append(CodeIssue(
                    severity='info',
                    category='style',
                    line=i,
                    description='Trailing whitespace',
                    suggestion='Remove trailing whitespace'
                ))
                
        # Language-specific checks
        if language == 'python':
            issues.extend(self._python_specific_checks(code))
        elif language == 'cpp':
            issues.extend(self._cpp_specific_checks(code))
            
        return issues
    
    def _python_specific_checks(self, code: str) -> List[CodeIssue]:
        """Python-specific static analysis"""
        issues = []
        
        # Check for missing docstrings
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        issues.append(CodeIssue(
                            severity='warning',
                            category='style',
                            line=node.lineno,
                            description=f'{node.__class__.__name__} missing docstring',
                            suggestion='Add docstring to document purpose and parameters'
                        ))
        except SyntaxError:
            pass
            
        return issues
    
    def _cpp_specific_checks(self, code: str) -> List[CodeIssue]:
        """C++ specific static analysis"""
        issues = []
        
        # Check for common C++ issues
        if 'using namespace std;' in code:
            issues.append(CodeIssue(
                severity='warning',
                category='style',
                line=1,
                description='Using namespace std in global scope',
                suggestion='Avoid "using namespace std;" in headers'
            ))
            
        return issues
    
    def generate_report(self, results: List[AnalysisResult], output_format: str = 'json') -> str:
        """Generate analysis report"""
        if output_format == 'json':
            return json.dumps([asdict(r) for r in results], indent=2)
        elif output_format == 'markdown':
            return self._generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_markdown_report(self, results: List[AnalysisResult]) -> str:
        """Generate markdown report"""
        report = ["# AI Code Quality Analysis Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for result in results:
            report.append(f"\n## {result.file_path}\n")
            report.append(f"- **Language**: {result.language}")
            report.append(f"- **Overall Score**: {result.overall_score:.1f}/100")
            report.append(f"- **AI Confidence**: {result.ai_confidence:.1f}%")
            
            if result.issues:
                report.append(f"\n### Issues Found ({len(result.issues)})\n")
                for issue in result.issues:
                    emoji = {'critical': '🔴', 'warning': '🟡', 'info': 'ℹ️'}.get(issue.severity, '•')
                    report.append(f"{emoji} **Line {issue.line}** [{issue.category}]: {issue.description}")
                    report.append(f"   - *Suggestion*: {issue.suggestion}")
            else:
                report.append("\n✅ No issues found!")
                
            report.append(f"\n### Metrics")
            for key, value in result.metrics.items():
                if key not in ['ai_indicators', 'ai_analysis']:
                    report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                    
        return '\n'.join(report)


def main():
    """CLI interface for the analyzer"""
    parser = argparse.ArgumentParser(description='AI Code Quality Analyzer')
    parser.add_argument('files', nargs='+', help='Code files to analyze')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI analysis')
    parser.add_argument('--output', choices=['json', 'markdown'], default='markdown',
                        help='Output format (default: markdown)')
    parser.add_argument('--save', help='Save report to file')
    
    args = parser.parse_args()
    
    analyzer = AICodeAnalyzer(use_ai=not args.no_ai)
    results = []
    
    for file_path in args.files:
        try:
            result = analyzer.analyze_file(file_path)
            results.append(result)
            logger.info(f"Successfully analyzed {file_path}")
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            
    if results:
        report = analyzer.generate_report(results, args.output)
        
        if args.save:
            with open(args.save, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.save}")
        else:
            print(report)


if __name__ == "__main__":
    main()