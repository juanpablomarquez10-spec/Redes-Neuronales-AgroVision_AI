from huggingface_hub import HfApi

print("Consultando las categorías del conjunto de datos...")

api = HfApi()

archivos = api.list_repo_files(
    repo_id="lorenzoxi/tomato-leaves-dataset",
    repo_type="dataset"
)

categorias = set()

for archivo in archivos:
    partes = archivo.split("/")

    if len(partes) >= 4 and partes[1] in ("train", "test", "valid"):
        categorias.add(partes[2])

print("\nCategorías encontradas:")

for categoria in sorted(categorias):
    print("-", categoria)

print(f"\nTotal de categorías: {len(categorias)}")