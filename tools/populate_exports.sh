root_dir=$(pwd)
src_dir="$root_dir/src"
exports_dir="$root_dir/exports"
if [ ! -d "$src_dir" ]; then
    echo "The 'src' directory could not be found."
    exit 1
fi

if [ ! -d "$exports_dir" ]; then
    mkdir "$exports_dir"
    echo "Created 'exports' directory."
fi

exports_file="$exports_dir/Exports.jl"
> "$exports_file"
echo -e "# This file is auto-generated. To edit, run 'tool/populate_exports.sh'\n" > "$exports_file"
for module_name in "$src_dir"/*; do
    if [ -d "$module_name" ]; then
        module_dir="$module_name"
        module_export_path="$module_dir/${module_name##*/}Exports.jl"
        main_export_path="$exports_dir/${module_name##*/}.jl"
        if [ -f "$module_export_path" ]; then
            export_content=$(cat "$module_export_path")
			echo -e "# This file is auto-generated. To edit, run 'tool/populate_exports.sh'\n" > "$main_export_path"
            echo "using .${module_name##*/}" >> "$main_export_path"
            echo -e "\n$export_content" >> "$main_export_path"
			echo "include(\"${module_name##*/}.jl\")" >> "$exports_file"
            echo "Generated export file for ${module_name##*/}"
        fi
    fi
done
