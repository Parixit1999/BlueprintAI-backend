"""File-manager operations: folder tree CRUD, file rename/move, browsing.

Folder location is purely organizational - a file keeps its extraction,
project/drawing assignment, and RAG behavior wherever it lives. Deleting a
folder deletes its subtree and the files inside (through FileService so
storage objects and chunks are cleaned up too), like a standard file manager.
"""
from app.exceptions import FileNotFound, UnsupportedFileType
from app.repositories import FileRepository, FolderRepository
from app.services.file_service import FileService


class FolderService:
    def __init__(self, folders: FolderRepository, files: FileRepository, file_service: FileService):
        self._folders = folders
        self._files = files
        self._file_service = file_service

    def _require(self, folder_id: str) -> dict:
        folder = self._folders.get(folder_id)
        if folder is None:
            raise FileNotFound("Folder not found")
        return folder

    def create(self, name: str, parent_id: str | None) -> dict:
        clean = name.strip()
        if not clean:
            raise UnsupportedFileType("Folder name cannot be empty.")
        if parent_id:
            self._require(parent_id)
        return self._folders.create(clean, parent_id)

    def browse(self, folder_id: str | None) -> dict:
        """One folder-manager view: current folder, breadcrumb path back to
        root, subfolders with counts, and the files directly inside."""
        folder = self._require(folder_id) if folder_id else None
        breadcrumbs: list[dict] = []
        cursor = folder
        while cursor is not None:
            breadcrumbs.insert(0, {"folder_id": cursor["folder_id"], "name": cursor["name"]})
            cursor = self._folders.get(cursor["parent_id"]) if cursor["parent_id"] else None
        return {
            "folder": folder,
            "breadcrumbs": breadcrumbs,
            "folders": self._folders.children(folder_id),
            "files": self._files.list_in_folder(folder_id),
        }

    def list_all(self) -> list[dict]:
        return self._folders.list_all()

    def rename(self, folder_id: str, name: str) -> dict:
        self._require(folder_id)
        clean = name.strip()
        if not clean:
            raise UnsupportedFileType("Folder name cannot be empty.")
        self._folders.rename(folder_id, clean)
        return self._folders.get(folder_id)

    def move(self, folder_id: str, parent_id: str | None) -> dict:
        self._require(folder_id)
        if parent_id:
            self._require(parent_id)
            # a folder cannot move into itself or its own subtree
            if parent_id in self._folders.subtree_ids(folder_id):
                raise UnsupportedFileType(
                    "A folder cannot be moved into itself or one of its subfolders."
                )
        self._folders.move(folder_id, parent_id)
        return self._folders.get(folder_id)

    def delete(self, folder_id: str) -> dict:
        """Recursive delete: every file in the subtree is removed properly
        (storage objects + chunks), then the folder rows cascade away."""
        self._require(folder_id)
        subtree = self._folders.subtree_ids(folder_id)
        file_ids = self._folders.file_ids_in(subtree)
        for fid in file_ids:
            self._file_service.delete_file(fid)
        self._folders.delete(folder_id)
        return {"deleted_folders": len(subtree), "deleted_files": len(file_ids)}

    # --- file operations ---

    def rename_file(self, file_id: str, filename: str) -> dict:
        if self._files.get(file_id) is None:
            raise FileNotFound("File not found")
        clean = filename.strip()
        if not clean:
            raise UnsupportedFileType("File name cannot be empty.")
        self._files.rename(file_id, clean)
        return {"file_id": file_id, "filename": clean}

    def move_file(self, file_id: str, folder_id: str | None) -> dict:
        if self._files.get(file_id) is None:
            raise FileNotFound("File not found")
        if folder_id:
            self._require(folder_id)
        self._files.move_to_folder(file_id, folder_id)
        return {"file_id": file_id, "folder_id": folder_id}
