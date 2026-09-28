#!/usr/bin/env bash

OUTPUT_FILE="project_map.txt"

echo "=== ARBORESCENCE DU PROJET ===" > "$OUTPUT_FILE"

if command -v tree &> /dev/null; then
    tree -I "build|.git|.vs|.vscode|bin|obj" >> "$OUTPUT_FILE"
else
    find . -maxdepth 4 -not -path '*/.*' -not -path './build*' -not -path './bin*' -not -path './obj*' >> "$OUTPUT_FILE"
fi

echo -e "\n==========================================" >> "$OUTPUT_FILE"
echo "=== CONTENU DES FICHIERS DE CODE ===" >> "$OUTPUT_FILE"
echo "==========================================" >> "$OUTPUT_FILE"

find . -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.c" \) \
    -not -path "*/build/*" \
    -not -path "*/.git/*" \
    -not -path "*/.vs/*" | while read -r file; do
    echo -e "\n--- DEBUT FICHIER: $file ---" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\n--- FIN FICHIER: $file ---" >> "$OUTPUT_FILE"
done

echo "Cartographie générée dans $OUTPUT_FILE."