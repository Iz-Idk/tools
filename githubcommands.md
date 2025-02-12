Aquí tienes tu documentación en un archivo Markdown bien organizado:  

### 📄 **Archivo:** `github_upload_guide.md`  

```md
# 📌 Documentación para Subir Código a GitHub

## 🚀 Primera vez subiendo código

Ejecuta los siguientes comandos en la terminal para inicializar un repositorio y subirlo a GitHub:

```sh
git init
git remote add origin https://github.com/username/repository-name.git
git add .
git commit -m "Initial commit with project files"
git push -u origin main
```

---

## 📂 Subiendo más archivos

Si necesitas agregar más archivos al repositorio, sigue estos pasos:

```sh
git checkout -b temporal  # Crear y cambiar a una nueva rama
git add .                 # Agregar archivos al área de preparación
git status                # Verificar el estado de los cambios
git commit -m "Descripción de los cambios"
git push -u origin temporal  # Subir la rama temporal a GitHub
```

---

## 🔗 Conectar el branch con `main`

Si el branch no está conectado correctamente con `main`, usa los siguientes comandos:

```sh
git remote set-url origin https://github.com/Iz-Idk/tools.git  # Configurar la URL remota
git pull --rebase origin main  # Traer los últimos cambios de main y aplicar rebase
git push -u origin main  # Subir cambios a main
```

---

## 🏷️ Manejo de ramas

Para asegurarte de que `main` y `master` están correctamente alineados, usa:

```sh
git checkout master           # Cambiar a la rama master
git branch main master -f     # Forzar main a ser igual que master
git checkout main             # Volver a la rama main
git push origin main -f       # Forzar el push de main a GitHub
```

---

✅ **¡Listo! Ahora tu código está correctamente subido y sincronizado en GitHub.**  
```

