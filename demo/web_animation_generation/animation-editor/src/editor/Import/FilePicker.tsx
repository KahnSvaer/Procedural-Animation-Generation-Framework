import { forwardRef } from "react";

interface FilePickerProps {
    onFileSelected: (file: File) => void;
}

const FilePicker = forwardRef<HTMLInputElement, FilePickerProps>(
    ({ onFileSelected }, ref) => {

        const handleChange = (
            event: React.ChangeEvent<HTMLInputElement>
        ) => {
            const file = event.target.files?.[0];

            if (!file) return;

            onFileSelected(file);

            event.target.value = "";
        };

        return (
            <input
                ref={ref}
                type="file"
                accept=".glb,.gltf"
                hidden
                onChange={handleChange}
            />
        );
    }
);

FilePicker.displayName = "FilePicker";

export default FilePicker;