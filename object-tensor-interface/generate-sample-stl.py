import torch
import os
import numpy as np

'''

FORMAT STL:

solid nazwa_obiektu
  facet normal nx ny nz
    outer loop
      vertex x1 y1 z1
      vertex x2 y2 z2
      vertex x3 y3 z3
    endloop
  endfacet
  ...
endsolid

czyli:

- definiujemy nazwę modelu
- dla każdej ściany obliczamy jej normalną (nx, ny, nz), i zapisujemy jako floaty rozdzielone spacją
- dla każdej ściany definiujemy outer loop, czyli ograniczenia płaszczyzny
- elementami outerloop są trzy wierzchołki, które zapisujemy tak samo jak normalną: vertex x y z, gdzie każda współrzędna to float, są rozdzielone spacją
- endloop, endfacet i nowy trójkąt tak samo - facet normal nx ny nz, outer loop...
- endsolid jako znak końca modelu

'''

# Definicja sześcianu jako modelu STL
cube_stl_model = [
    # Pierwsza ściana (trójkąt 1)
    [
        [-1.0, -1.0, -1.0],  # Pierwszy wierzchołek (x, y, z)
        [1.0, -1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [-1.0, 1.0, -1.0]    # Trzeci wierzchołek (x, y, z)
    ],
    # Pierwsza ściana (trójkąt 2)
    [
        [-1.0, 1.0, -1.0],   # Pierwszy wierzchołek (x, y, z)
        [1.0, -1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [1.0, 1.0, -1.0]     # Trzeci wierzchołek (x, y, z)
    ],
    # Druga ściana (trójkąt 1)
    [
        [-1.0, -1.0, -1.0],  # Pierwszy wierzchołek (x, y, z)
        [-1.0, 1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [-1.0, -1.0, 1.0]    # Trzeci wierzchołek (x, y, z)
    ],
    # Druga ściana (trójkąt 2)
    [
        [-1.0, -1.0, 1.0],   # Pierwszy wierzchołek (x, y, z)
        [-1.0, 1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [-1.0, 1.0, 1.0]     # Trzeci wierzchołek (x, y, z)
    ],
    # Trzecia ściana (trójkąt 1)
    [
        [-1.0, -1.0, -1.0],  # Pierwszy wierzchołek (x, y, z)
        [-1.0, -1.0, 1.0],   # Drugi wierzchołek (x, y, z)
        [1.0, -1.0, -1.0]    # Trzeci wierzchołek (x, y, z)
    ],
    # Trzecia ściana (trójkąt 2)
    [
        [1.0, -1.0, -1.0],   # Pierwszy wierzchołek (x, y, z)
        [-1.0, -1.0, 1.0],   # Drugi wierzchołek (x, y, z)
        [1.0, -1.0, 1.0]     # Trzeci wierzchołek (x, y, z)
    ],
    # Czwarta ściana (trójkąt 1)
    [
        [-1.0, -1.0, -1.0],  # Pierwszy wierzchołek (x, y, z)
        [1.0, -1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [-1.0, -1.0, 1.0]    # Trzeci wierzchołek (x, y, z)
    ],
    # Czwarta ściana (trójkąt 2)
    [
        [-1.0, -1.0, 1.0],   # Pierwszy wierzchołek (x, y, z)
        [1.0, -1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [1.0, -1.0, 1.0]     # Trzeci wierzchołek (x, y, z)
    ],
    # Piąta ściana (trójkąt 1)
    [
        [-1.0, -1.0, -1.0],  # Pierwszy wierzchołek (x, y, z)
        [-1.0, 1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [-1.0, -1.0, 1.0]    # Trzeci wierzchołek (x, y, z)
    ],
    # Piąta ściana (trójkąt 2)
    [
        [-1.0, -1.0, 1.0],   # Pierwszy wierzchołek (x, y, z)
        [-1.0, 1.0, -1.0],   # Drugi wierzchołek (x, y, z)
        [-1.0, 1.0, 1.0]     # Trzeci wierzchołek (x, y, z)
    ],
    # Szósta ściana (trójkąt 1)
    [
        [-1.0, 1.0, -1.0],   # Pierwszy wierzchołek (x, y, z)
        [1.0, 1.0, -1.0],    # Drugi wierzchołek (x, y, z)
        [-1.0, 1.0, 1.0]     # Trzeci wierzchołek (x, y, z)
    ],
    # Szósta ściana (trójkąt 2)
    [
        [-1.0, 1.0, 1.0],    # Pierwszy wierzchołek (x, y, z)
        [1.0, 1.0, -1.0],    # Drugi wierzchołek (x, y, z)
        [1.0, 1.0, 1.0]      # Trzeci wierzchołek (x, y, z)
    ]
]

# Konwersja na tensor PyTorch
tensor_cube_stl_model = torch.tensor(cube_stl_model, dtype=torch.float32)

# Wymiary tensora tensor_cube_stl_model będą (12, 3, 3) odpowiadające 12 trójkątom (bo po 2 na ścianę), z których każdy ma 3 wierzchołki po 3 współrzędne (x, y, z)

'''

Obliczenie normalnej trójkąta: format stl przyjmuje normalne trójkątów. Musimy je policzyć: tu jako iloczyn wektorowy boków, można też jakoś inaczej to liczyć, ale jest prościej to tak zapisać

'''

def compute_triangle_normal(vertex1, vertex2, vertex3):
    # Oblicz dwa wektory boków trójkąta
    edge1 = vertex2 - vertex1
    edge2 = vertex3 - vertex1

    # Oblicz iloczyn wektorowy dwóch boków trójkąta
    normal = torch.cross(edge1, edge2)

    # Zwróć znormalizowaną normalną trójkąta (o długości równiej polu trójkąta)
    return normal / torch.norm(normal)


def tensor_to_stl_ascii(tensor_stl_model, output_filename):
    # Uzyskaj ścieżkę do folderu roboczego
    current_dir = os.getcwd()
    
    # Łącz ścieżkę folderu roboczego z nazwą pliku STL
    output_path = os.path.join(current_dir, output_filename)

    # Otwórz plik STL do zapisu w trybie ASCII
    with open(output_path, 'w') as stl_file:
        # Nagłówek pliku STL (opcjonalny)
        stl_file.write("solid cube\n")

        # Przetwarzanie tensora z trójkątnymi powierzchniami
        for triangle in tensor_stl_model:
            vertex1 = triangle[0]
            vertex2 = triangle[1]
            vertex3 = triangle[2]
            
            normal = compute_triangle_normal(vertex1, vertex2, vertex3)
            # Zapisz normalną trójkąta
            stl_file.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")

            # Przejdź przez wierzchołki trójkąta i zapisz je
            stl_file.write("    outer loop\n")
            for vertex in triangle:
                stl_file.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            stl_file.write("    endloop\n")
            stl_file.write("  endfacet\n")

        # Stopka pliku STL
        stl_file.write("endsolid cube\n")

# Nazwa pliku do zapisu
output_filename = "cube.stl"

# Konwertuj tensor na plik STL w trybie ASCII i zapisz w folderze roboczym
tensor_to_stl_ascii(tensor_cube_stl_model, output_filename)