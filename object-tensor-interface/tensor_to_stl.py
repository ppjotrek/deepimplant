import os
import torch

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