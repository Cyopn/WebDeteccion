from pathlib import Path


def describe(line: str) -> str:
    t = line.strip()
    if t == '':
        return 'Línea vacía.'
    if t.startswith('<!--'):
        return 'Comentario HTML.'
    if t.startswith('//'):
        body = t.lstrip('/').strip()
        return f'Comentario: {body}' if body else 'Comentario.'
    if t.startswith('/*'):
        return 'Inicio o parte de comentario en bloque.'
    if t.endswith('*/'):
        return 'Fin de comentario en bloque.'
    if t.startswith('import '):
        return f'Importa módulo ({t}).'
    if t.startswith('const '):
        return f'Define constante ({t}).'
    if t.startswith('let '):
        return f'Define variable ({t}).'
    if t.startswith('function '):
        name = t.split()[1].split('(')[0]
        return f'Define la función {name}.'
    if t.startswith('async function '):
        name = t.split()[2].split('(')[0]
        return f'Define la función asíncrona {name}.'
    if t.startswith('class '):
        name = t.split()[1].split('{')[0]
        return f'Declara la clase {name}.'
    if t.startswith('if '):
        return f'Condicional if ({t[3:].strip()}).'
    if t.startswith('else if '):
        return f'Condicional else if ({t[8:].strip()}).'
    if t.startswith('else'):
        return 'Bloque else.'
    if t.startswith('for '):
        return f'Bucle for ({t[4:].strip()}).'
    if t.startswith('while '):
        return f'Bucle while ({t[6:].strip()}).'
    if t.startswith('return'):
        body = t[6:].strip()
        return f'Retorna {body or "un valor"}.'
    if t.startswith('try'):
        return 'Inicio de bloque try.'
    if t.startswith('catch'):
        return 'Bloque catch.'
    if t.startswith('finally'):
        return 'Bloque finally.'
    if t.startswith('switch'):
        return f'Inicio de switch ({t}).'
    if t.startswith('case '):
        return f'Caso de switch ({t}).'
    if t.startswith('default'):
        return 'Caso default de switch.'
    if t.startswith('<!doctype') or t.startswith('<!DOCTYPE'):
        return 'Declaración doctype.'
    if t.startswith('<html'):
        return 'Etiqueta html de apertura.'
    if t.startswith('</html'):
        return 'Etiqueta html de cierre.'
    if t.startswith('<head'):
        return 'Sección head de apertura.'
    if t.startswith('</head'):
        return 'Sección head de cierre.'
    if t.startswith('<body'):
        return 'Etiqueta body de apertura.'
    if t.startswith('</body'):
        return 'Etiqueta body de cierre.'
    if t.startswith('<script'):
        return 'Etiqueta script de apertura.'
    if t.startswith('</script'):
        return 'Etiqueta script de cierre.'
    if t.startswith('<'):
        return f'Etiqueta/elemento HTML ({t}).'
    if t.startswith('[') and '](' in t:
        return f'Enlace markdown ({t}).'
    return f'Código: {t}'


def build_entries(path: str, max_lines: int | None = None):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if max_lines is not None:
        lines = lines[:max_lines]
    entries = []
    for idx, line in enumerate(lines, 1):
        desc = describe(line)
        entries.append(
            f"- L{idx}: [{path}](client/{Path(path).name}#L{idx}) - {desc}")
    return entries


def main():
    entries = []
    entries.extend(build_entries('client/main.js', max_lines=898))
    entries.append('')
    entries.extend(build_entries('client/index.html', max_lines=488))
    Path('doc_client.md').write_text(
        '\n'.join(entries), encoding='utf-8')
    print('doc_client.md generado con', len(entries), 'líneas')


if __name__ == '__main__':
    main()
