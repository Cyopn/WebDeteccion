from pathlib import Path


def describe(s: str) -> str:
    t = s.strip()
    if t == '':
        return 'Línea vacía.'
    if t.startswith('#'):
        body = t.lstrip('#').strip()
        return f'Comentario: {body}' if body else 'Comentario.'
    if t.startswith('"""') or t.startswith("'''"):
        return 'Inicio o fin de cadena multilínea.'
    if t.startswith('from '):
        return f'Importa módulo/atributos ({t}).'
    if t.startswith('import '):
        return f'Importa módulo ({t}).'
    if t.startswith('class '):
        name = t.split()[1].split('(')[0].strip(':')
        return f'Declara la clase {name}.'
    if t.startswith('def '):
        name = t.split()[1].split('(')[0]
        return f'Define la función {name}.'
    if t.startswith('@'):
        return f'Decorador {t}.'
    if t.startswith('try'):
        return 'Inicio de bloque try para manejo de errores.'
    if t.startswith('except'):
        return f'Manejo de excepción ({t}).'
    if t.startswith('finally'):
        return 'Bloque finally.'
    if t.startswith('if '):
        return f'Condicional if ({t[3:].strip()}).'
    if t.startswith('elif '):
        return f'Condicional elif ({t[5:].strip()}).'
    if t.startswith('else'):
        return 'Bloque else.'
    if t.startswith('for '):
        return f'Bucle for ({t[4:].strip()}).'
    if t.startswith('while '):
        return f'Bucle while ({t[6:].strip()}).'
    if t.startswith('with '):
        return f'Context manager ({t[5:].strip()}).'
    if t.startswith('return'):
        body = t[6:].strip()
        return f'Retorna {body or "un valor"}.'
    if t.startswith('yield'):
        body = t[5:].strip()
        return f'Genera {body or "un valor"}.'
    if t.startswith('raise '):
        return f'Lanza excepción {t[6:].strip()}.'
    if t.startswith('pass'):
        return 'Sentencia pass (sin acción).'
    if t.startswith('global '):
        return f'Declara uso de variables globales ({t[7:].strip()}).'
    if t.startswith('nonlocal '):
        return f'Declara uso de variables nonlocal ({t[9:].strip()}).'
    return f'Código: {t}'


def main():
    lines = Path('app.py').read_text(encoding='utf-8').splitlines()
    output_lines = [
        f"- L{idx}: [app.py](app.py#L{idx}) - {describe(line)}"
        for idx, line in enumerate(lines, 1)
    ]
    Path('doc_app.md').write_text(
        '\n'.join(output_lines), encoding='utf-8')
    print(f"doc_app.md generado con {len(output_lines)} líneas")


if __name__ == '__main__':
    main()
