# Flujo de colaboracion de GeoRisk AI

## Ramas

- `main`: entrega estable. Solo recibe Pull Requests desde `develop`.
- `develop`: integracion y pruebas. Solo recibe Pull Requests desde ramas de trabajo.
- `feat/<tarea>`: funcionalidad o mejora nueva.
- `fix/<error>`: correccion puntual.

Cada rama de trabajo debe nacer desde `develop` actualizado:

```bash
git switch develop
git pull origin develop
git switch -c feat/nombre-tarea
```

## Commits y Pull Requests

Usamos Conventional Commits: `feat:`, `fix:`, `docs:`, `test:` o `chore:`. Cada
integrante hace sus commits con su propia cuenta y abre su PR hacia `develop`.
La descripcion debe incluir evidencia de pruebas y, cuando corresponda, metricas
antes y despues. Otra persona del equipo debe revisar el PR.

Antes de subir cambios:

```bash
python -m pytest -q
cd frontend
npm ci
npm run build
```

## Versiones

Las entregas fusionadas de `develop` a `main` se etiquetan con SemVer:

- `MAJOR`: cambio incompatible o nuevo hito.
- `MINOR`: funcionalidad compatible.
- `PATCH`: correccion compatible.

El administrador crea el tag anotado y publica el Release en GitHub. No se deben
versionar el CSV generado, archivos `.joblib`, `artifacts/`, secretos ni entornos.
