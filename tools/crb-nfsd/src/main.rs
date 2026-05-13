//! Userspace NFSv3 server for the claude-remote-bash filesystem-mount feature.
//!
//! Serves a local directory as NFSv3 over TCP. Spawned by the claude-remote-bash
//! daemon and accessed by macOS `mount_nfs` via a port-forward tunnel.
//!
//! Adapted from `xetdata/nfsserve`'s `examples/mirrorfs.rs` (BSD-3-Clause).
//! Local additions:
//!   - clap-based CLI (`--listen`, `--root`, `--readonly`, `--block-prefix`)
//!   - `--block-prefix` enforcement in `lookup` (canonicalized via parent)
//!   - `--readonly` enforcement via `capabilities()`
//!   - macOS pollution-name refusal (lookup/readdir/create/remove/rename)
//!   - `LISTEN_PORT=<n>\nREADY` stdout signal for parent-process port capture
//!
//! Upstream license preserved at `LICENSE-UPSTREAM`.

use std::collections::{BTreeSet, HashMap};
use std::ffi::{OsStr, OsString};
use std::fs::Metadata;
use std::io::SeekFrom;
use std::ops::Bound;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use clap::Parser;
use intaglio::osstr::SymbolTable;
use intaglio::Symbol;
use nfsserve::fs_util::*;
use nfsserve::nfs::*;
use nfsserve::tcp::{NFSTcp, NFSTcpListener};
use nfsserve::vfs::{DirEntry, NFSFileSystem, ReadDirResult, VFSCapabilities};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tracing::debug;

/// macOS metadata filenames that Finder, Spotlight, and Quick Look auto-create
/// on any browsed folder. Filtering these out of every namespace operation
/// (lookup, readdir, create, remove, rename) keeps Finder/Spotlight from
/// polluting the peer's filesystem when the mount is opened in GUI.
fn is_pollution_pattern(name: &[u8]) -> bool {
    if name.starts_with(b"._") {
        // AppleDouble sidecar files for extended attributes / resource forks.
        return true;
    }
    matches!(
        name,
        b".DS_Store"
            | b".Spotlight-V100"
            | b".Trashes"
            | b".fseventsd"
            | b".TemporaryItems"
            | b".DocumentRevisions-V100"
            | b"Icon\r" // literal 0x0D byte — the macOS "Custom Folder Icon" filename
    )
}

#[derive(Debug, Clone)]
struct FSEntry {
    name: Vec<Symbol>,
    fsmeta: fattr3,
    /// metadata when building the children list
    children_meta: fattr3,
    children: Option<BTreeSet<fileid3>>,
}

#[derive(Debug)]
struct FSMap {
    root: PathBuf,
    next_fileid: AtomicU64,
    intern: SymbolTable,
    id_to_path: HashMap<fileid3, FSEntry>,
    path_to_id: HashMap<Vec<Symbol>, fileid3>,
}

enum RefreshResult {
    /// The fileid was deleted
    Delete,
    /// The fileid needs to be reloaded. mtime has been updated, caches
    /// need to be evicted.
    Reload,
    /// Nothing has changed
    Noop,
}

impl FSMap {
    fn new(root: PathBuf) -> FSMap {
        // create root entry
        let root_entry = FSEntry {
            name: Vec::new(),
            fsmeta: metadata_to_fattr3(1, &root.metadata().unwrap()),
            children_meta: metadata_to_fattr3(1, &root.metadata().unwrap()),
            children: None,
        };
        FSMap {
            root,
            next_fileid: AtomicU64::new(1),
            intern: SymbolTable::new(),
            id_to_path: HashMap::from([(0, root_entry)]),
            path_to_id: HashMap::from([(Vec::new(), 0)]),
        }
    }
    async fn sym_to_path(&self, symlist: &[Symbol]) -> PathBuf {
        let mut ret = self.root.clone();
        for i in symlist.iter() {
            ret.push(self.intern.get(*i).unwrap());
        }
        ret
    }

    async fn sym_to_fname(&self, symlist: &[Symbol]) -> OsString {
        if let Some(x) = symlist.last() {
            self.intern.get(*x).unwrap().into()
        } else {
            "".into()
        }
    }

    fn collect_all_children(&self, id: fileid3, ret: &mut Vec<fileid3>) {
        ret.push(id);
        if let Some(entry) = self.id_to_path.get(&id) {
            if let Some(ref ch) = entry.children {
                for i in ch.iter() {
                    self.collect_all_children(*i, ret);
                }
            }
        }
    }

    fn delete_entry(&mut self, id: fileid3) {
        let mut children = Vec::new();
        self.collect_all_children(id, &mut children);
        for i in children.iter() {
            if let Some(ent) = self.id_to_path.remove(i) {
                self.path_to_id.remove(&ent.name);
            }
        }
    }

    fn find_entry(&self, id: fileid3) -> Result<FSEntry, nfsstat3> {
        Ok(self.id_to_path.get(&id).ok_or(nfsstat3::NFS3ERR_NOENT)?.clone())
    }
    fn find_entry_mut(&mut self, id: fileid3) -> Result<&mut FSEntry, nfsstat3> {
        self.id_to_path.get_mut(&id).ok_or(nfsstat3::NFS3ERR_NOENT)
    }
    async fn find_child(&self, id: fileid3, filename: &[u8]) -> Result<fileid3, nfsstat3> {
        let mut name = self.id_to_path.get(&id).ok_or(nfsstat3::NFS3ERR_NOENT)?.name.clone();
        name.push(
            self.intern
                .check_interned(OsStr::from_bytes(filename))
                .ok_or(nfsstat3::NFS3ERR_NOENT)?,
        );
        Ok(*self.path_to_id.get(&name).ok_or(nfsstat3::NFS3ERR_NOENT)?)
    }
    async fn refresh_entry(&mut self, id: fileid3) -> Result<RefreshResult, nfsstat3> {
        let entry = self.id_to_path.get(&id).ok_or(nfsstat3::NFS3ERR_NOENT)?.clone();
        let path = self.sym_to_path(&entry.name).await;
        if !exists_no_traverse(&path) {
            self.delete_entry(id);
            debug!("Deleting entry A {:?}: {:?}. Ent: {:?}", id, path, entry);
            return Ok(RefreshResult::Delete);
        }

        let meta = tokio::fs::symlink_metadata(&path).await.map_err(|_| nfsstat3::NFS3ERR_IO)?;
        let meta = metadata_to_fattr3(id, &meta);
        if !fattr3_differ(&meta, &entry.fsmeta) {
            return Ok(RefreshResult::Noop);
        }
        // If we get here we have modifications
        if entry.fsmeta.ftype as u32 != meta.ftype as u32 {
            // if the file type changed ex: file->dir or dir->file
            // really the entire file has been replaced.
            // we expire the entire id
            debug!("File Type Mismatch FT {:?} : {:?} vs {:?}", id, entry.fsmeta.ftype, meta.ftype);
            debug!("File Type Mismatch META {:?} : {:?} vs {:?}", id, entry.fsmeta, meta);
            self.delete_entry(id);
            debug!("Deleting entry B {:?}: {:?}. Ent: {:?}", id, path, entry);
            return Ok(RefreshResult::Delete);
        }
        // inplace modification.
        // update metadata
        self.id_to_path.get_mut(&id).unwrap().fsmeta = meta;
        debug!("Reloading entry {:?}: {:?}. Ent: {:?}", id, path, entry);
        Ok(RefreshResult::Reload)
    }
    async fn refresh_dir_list(&mut self, id: fileid3) -> Result<(), nfsstat3> {
        let entry = self.id_to_path.get(&id).ok_or(nfsstat3::NFS3ERR_NOENT)?.clone();
        // if there are children and the metadata did not change
        if entry.children.is_some() && !fattr3_differ(&entry.children_meta, &entry.fsmeta) {
            return Ok(());
        }
        if !matches!(entry.fsmeta.ftype, ftype3::NF3DIR) {
            return Ok(());
        }
        let mut cur_path = entry.name.clone();
        let path = self.sym_to_path(&entry.name).await;
        let mut new_children: Vec<u64> = Vec::new();
        debug!("Relisting entry {:?}: {:?}. Ent: {:?}", id, path, entry);
        if let Ok(mut listing) = tokio::fs::read_dir(&path).await {
            while let Some(entry) = listing.next_entry().await.map_err(|_| nfsstat3::NFS3ERR_IO)? {
                let sym = self.intern.intern(entry.file_name()).unwrap();
                cur_path.push(sym);
                let meta = entry.metadata().await.unwrap();
                let next_id = self.create_entry(&cur_path, meta).await;
                new_children.push(next_id);
                cur_path.pop();
            }
            self.id_to_path.get_mut(&id).ok_or(nfsstat3::NFS3ERR_NOENT)?.children =
                Some(BTreeSet::from_iter(new_children.into_iter()));
        }

        Ok(())
    }

    async fn create_entry(&mut self, fullpath: &[Symbol], meta: Metadata) -> fileid3 {
        let next_id = if let Some(chid) = self.path_to_id.get(fullpath) {
            if let Some(chent) = self.id_to_path.get_mut(chid) {
                chent.fsmeta = metadata_to_fattr3(*chid, &meta);
            }
            *chid
        } else {
            // path does not exist
            let next_id = self.next_fileid.fetch_add(1, Ordering::Relaxed);
            let metafattr = metadata_to_fattr3(next_id, &meta);
            let new_entry = FSEntry {
                name: fullpath.to_vec(),
                fsmeta: metafattr,
                children_meta: metafattr,
                children: None,
            };
            debug!("creating new entry {:?}: {:?}", next_id, meta);
            self.id_to_path.insert(next_id, new_entry);
            self.path_to_id.insert(fullpath.to_vec(), next_id);
            next_id
        };
        next_id
    }
}
#[derive(Debug)]
pub struct MirrorFS {
    fsmap: tokio::sync::Mutex<FSMap>,
    readonly: bool,
    /// Canonical absolute path prefixes that must not be exposed. Hit in `lookup`.
    block_prefixes: Vec<PathBuf>,
}

/// Enumeration for the create_fs_object method
enum CreateFSObject {
    /// Creates a directory
    Directory,
    /// Creates a file with a set of attributes
    File(sattr3),
    /// Creates an exclusive file with a set of attributes
    Exclusive,
    /// Creates a symlink with a set of attributes to a target location
    Symlink((sattr3, nfspath3)),
}
impl MirrorFS {
    pub fn new(root: PathBuf, readonly: bool, block_prefixes: Vec<PathBuf>) -> MirrorFS {
        MirrorFS {
            fsmap: tokio::sync::Mutex::new(FSMap::new(root)),
            readonly,
            block_prefixes,
        }
    }

    /// True if `path` (already canonical or close to it) lies under any
    /// configured block prefix. Used to mask off our own infra directories
    /// (e.g., `$HOME/.crb/`) from the served filesystem.
    fn is_blocked(&self, path: &Path) -> bool {
        self.block_prefixes.iter().any(|prefix| path.starts_with(prefix))
    }

    /// creates a FS object in a given directory and of a given type
    /// Updates as much metadata as we can in-place
    async fn create_fs_object(
        &self,
        dirid: fileid3,
        objectname: &filename3,
        object: &CreateFSObject,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        // Refuse Apple metadata sidecar names so Finder/Spotlight/Quick Look
        // can't pollute the peer's filesystem with `.DS_Store`, `._*`, etc.
        // EACCES (not EPERM) — per RFC 1813 EACCES is the policy-refused status
        // Finder handles gracefully; EPERM can trigger elevated-retry paths.
        if is_pollution_pattern(objectname) {
            return Err(nfsstat3::NFS3ERR_ACCES);
        }
        let mut fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(dirid)?;
        let mut path = fsmap.sym_to_path(&ent.name).await;
        let objectname_osstr = OsStr::from_bytes(objectname).to_os_string();
        path.push(&objectname_osstr);

        match object {
            CreateFSObject::Directory => {
                debug!("mkdir {:?}", path);
                if exists_no_traverse(&path) {
                    return Err(nfsstat3::NFS3ERR_EXIST);
                }
                tokio::fs::create_dir(&path).await.map_err(|_| nfsstat3::NFS3ERR_IO)?;
            },
            CreateFSObject::File(setattr) => {
                debug!("create {:?}", path);
                let file = std::fs::File::create(&path).map_err(|_| nfsstat3::NFS3ERR_IO)?;
                let _ = file_setattr(&file, setattr).await;
            },
            CreateFSObject::Exclusive => {
                debug!("create exclusive {:?}", path);
                let _ = std::fs::File::options()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(|_| nfsstat3::NFS3ERR_EXIST)?;
            },
            CreateFSObject::Symlink((_, target)) => {
                debug!("symlink {:?} {:?}", path, target);
                if exists_no_traverse(&path) {
                    return Err(nfsstat3::NFS3ERR_EXIST);
                }
                tokio::fs::symlink(OsStr::from_bytes(target), &path)
                    .await
                    .map_err(|_| nfsstat3::NFS3ERR_IO)?;
                // we do not set attributes on symlinks
            },
        }

        let _ = fsmap.refresh_entry(dirid).await;

        let sym = fsmap.intern.intern(objectname_osstr).unwrap();
        let mut name = ent.name.clone();
        name.push(sym);
        let meta = path.symlink_metadata().map_err(|_| nfsstat3::NFS3ERR_IO)?;
        let fileid = fsmap.create_entry(&name, meta.clone()).await;

        // update the children list
        if let Some(ref mut children) = fsmap.id_to_path.get_mut(&dirid).ok_or(nfsstat3::NFS3ERR_NOENT)?.children {
            children.insert(fileid);
        }
        Ok((fileid, metadata_to_fattr3(fileid, &meta)))
    }
}

#[async_trait]
impl NFSFileSystem for MirrorFS {
    fn root_dir(&self) -> fileid3 {
        0
    }
    fn capabilities(&self) -> VFSCapabilities {
        if self.readonly {
            VFSCapabilities::ReadOnly
        } else {
            VFSCapabilities::ReadWrite
        }
    }

    async fn lookup(&self, dirid: fileid3, filename: &filename3) -> Result<fileid3, nfsstat3> {
        // Apple pollution sidecars are namespace-invisible — refuse before
        // touching the FS or the cache. NOENT (not ACCES) so client sees the
        // names as nonexistent, consistent with readdir filtering.
        if is_pollution_pattern(filename) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }

        let mut fsmap = self.fsmap.lock().await;

        // Resolve the candidate path up-front so we can (a) refuse blocked
        // prefixes before exposing anything and (b) reuse it for the
        // negative-lookup fast path below.
        let dirent = fsmap.find_entry(dirid)?;
        let mut path = fsmap.sym_to_path(&dirent.name).await;
        let objectname_osstr = OsStr::from_bytes(filename).to_os_string();
        path.push(&objectname_osstr);

        // Resolve symlinks in the parent chain so `is_blocked` catches the case
        // where a peer has a symlink inside the served tree pointing at blocked
        // territory (e.g., `~/aaa → /Users/chris/.crb`). Canonicalize the
        // PARENT (which always exists — we just looked it up via find_entry)
        // and re-join the literal leaf. This holds even when the leaf doesn't
        // exist (negative lookup under a symlinked ancestor). Async because the
        // rest of the file uses tokio::fs::* for I/O — keep the runtime worker
        // unblocked.
        //
        // Scope of defense: catches symlinks IN the ancestor chain. Does NOT
        // catch a leaf that is itself a symlink whose target lives under a
        // blocked prefix — macOS NFS clients resolve readlinks client-side, so
        // the follow-up lookups against the resolved target go through their
        // own parent-canonicalize on each segment. For stronger isolation use a
        // narrowly-scoped `--root` rather than relying on `--block-prefix`.
        let candidate = match path.parent() {
            Some(parent) => match tokio::fs::canonicalize(parent).await {
                Ok(canon_parent) => canon_parent.join(&objectname_osstr),
                Err(_) => path.clone(),
            },
            None => path.clone(),
        };
        if self.is_blocked(&candidate) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }

        if let Ok(id) = fsmap.find_child(dirid, filename).await {
            if fsmap.id_to_path.contains_key(&id) {
                return Ok(id);
            }
        }
        // Optimize for negative lookups: if the file doesn't exist on disk, bail.
        if !exists_no_traverse(&path) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        // ok the file actually exists.
        // that means something changed under me probably.
        // refresh.

        if let RefreshResult::Delete = fsmap.refresh_entry(dirid).await? {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        let _ = fsmap.refresh_dir_list(dirid).await;

        fsmap.find_child(dirid, filename).await
    }

    async fn getattr(&self, id: fileid3) -> Result<fattr3, nfsstat3> {
        let mut fsmap = self.fsmap.lock().await;
        if let RefreshResult::Delete = fsmap.refresh_entry(id).await? {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        let ent = fsmap.find_entry(id)?;
        let path = fsmap.sym_to_path(&ent.name).await;
        debug!("Stat {:?}: {:?}", path, ent);
        Ok(ent.fsmeta)
    }

    async fn read(&self, id: fileid3, offset: u64, count: u32) -> Result<(Vec<u8>, bool), nfsstat3> {
        let fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(id)?;
        let path = fsmap.sym_to_path(&ent.name).await;
        drop(fsmap);
        let mut f = File::open(&path).await.or(Err(nfsstat3::NFS3ERR_NOENT))?;
        let len = f.metadata().await.or(Err(nfsstat3::NFS3ERR_NOENT))?.len();
        let mut start = offset;
        // checked_add: real NFSv3 clients cap `count` at ~64KB so overflow is
        // theoretical, but a malformed tunnel-claim could craft `offset + count`
        // to wrap past u64::MAX. Refuse explicitly.
        let mut end = offset
            .checked_add(count as u64)
            .ok_or(nfsstat3::NFS3ERR_INVAL)?;
        let eof = end >= len;
        if start >= len {
            start = len;
        }
        if end > len {
            end = len;
        }
        f.seek(SeekFrom::Start(start)).await.or(Err(nfsstat3::NFS3ERR_IO))?;
        let mut buf = vec![0; (end - start) as usize];
        f.read_exact(&mut buf).await.or(Err(nfsstat3::NFS3ERR_IO))?;
        Ok((buf, eof))
    }

    async fn readdir(
        &self,
        dirid: fileid3,
        start_after: fileid3,
        max_entries: usize,
    ) -> Result<ReadDirResult, nfsstat3> {
        let mut fsmap = self.fsmap.lock().await;
        fsmap.refresh_entry(dirid).await?;
        fsmap.refresh_dir_list(dirid).await?;

        let entry = fsmap.find_entry(dirid)?;
        if !matches!(entry.fsmeta.ftype, ftype3::NF3DIR) {
            return Err(nfsstat3::NFS3ERR_NOTDIR);
        }
        debug!("readdir({:?}, {:?})", entry, start_after);
        // we must have children here
        let children = entry.children.ok_or(nfsstat3::NFS3ERR_IO)?;

        let mut ret = ReadDirResult {
            entries: Vec::new(),
            end: false,
        };

        let range_start = if start_after > 0 {
            Bound::Excluded(start_after)
        } else {
            Bound::Unbounded
        };

        let remaining_length = children.range((range_start, Bound::Unbounded)).count();
        let path = fsmap.sym_to_path(&entry.name).await;
        debug!("path: {:?}", path);
        debug!("children len: {:?}", children.len());
        debug!("remaining_len : {:?}", remaining_length);
        // skipped == hidden-from-client entries (blocked-prefix or pollution-pattern).
        // End-of-stream is reached when entries_returned + skipped == remaining_length.
        let mut skipped = 0_usize;
        for i in children.range((range_start, Bound::Unbounded)) {
            let fileid = *i;
            let fileent = fsmap.find_entry(fileid)?;
            let name = fsmap.sym_to_fname(&fileent.name).await;
            if is_pollution_pattern(name.as_bytes()) {
                skipped += 1;
                continue;
            }
            let child_path = fsmap.sym_to_path(&fileent.name).await;
            if self.is_blocked(&child_path) {
                skipped += 1;
                continue;
            }
            debug!("\t --- {:?} {:?}", fileid, name);
            ret.entries.push(DirEntry {
                fileid,
                name: name.as_bytes().into(),
                attr: fileent.fsmeta,
            });
            if ret.entries.len() >= max_entries {
                break;
            }
        }
        if ret.entries.len() + skipped == remaining_length {
            ret.end = true;
        }
        debug!("readdir_result:{:?}", ret);

        Ok(ret)
    }

    async fn setattr(&self, id: fileid3, setattr: sattr3) -> Result<fattr3, nfsstat3> {
        let mut fsmap = self.fsmap.lock().await;
        let entry = fsmap.find_entry(id)?;
        let path = fsmap.sym_to_path(&entry.name).await;
        path_setattr(&path, &setattr).await?;

        // I have to lookup a second time to update
        let metadata = path.symlink_metadata().or(Err(nfsstat3::NFS3ERR_IO))?;
        if let Ok(entry) = fsmap.find_entry_mut(id) {
            entry.fsmeta = metadata_to_fattr3(id, &metadata);
        }
        Ok(metadata_to_fattr3(id, &metadata))
    }
    async fn write(&self, id: fileid3, offset: u64, data: &[u8]) -> Result<fattr3, nfsstat3> {
        let fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(id)?;
        let path = fsmap.sym_to_path(&ent.name).await;
        drop(fsmap);
        debug!("write to init {:?}", path);
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .await
            .map_err(|e| {
                debug!("Unable to open {:?}", e);
                nfsstat3::NFS3ERR_IO
            })?;
        f.seek(SeekFrom::Start(offset)).await.map_err(|e| {
            debug!("Unable to seek {:?}", e);
            nfsstat3::NFS3ERR_IO
        })?;
        f.write_all(data).await.map_err(|e| {
            debug!("Unable to write {:?}", e);
            nfsstat3::NFS3ERR_IO
        })?;
        debug!("write to {:?} {:?} {:?}", path, offset, data.len());
        let _ = f.flush().await;
        let _ = f.sync_all().await;
        let meta = f.metadata().await.or(Err(nfsstat3::NFS3ERR_IO))?;
        Ok(metadata_to_fattr3(id, &meta))
    }

    async fn create(
        &self,
        dirid: fileid3,
        filename: &filename3,
        setattr: sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        self.create_fs_object(dirid, filename, &CreateFSObject::File(setattr)).await
    }

    async fn create_exclusive(&self, dirid: fileid3, filename: &filename3) -> Result<fileid3, nfsstat3> {
        Ok(self.create_fs_object(dirid, filename, &CreateFSObject::Exclusive).await?.0)
    }

    async fn remove(&self, dirid: fileid3, filename: &filename3) -> Result<(), nfsstat3> {
        // Apple pollution sidecars are namespace-invisible — report NOENT,
        // consistent with lookup() and readdir() filtering.
        if is_pollution_pattern(filename) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        let mut fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(dirid)?;
        let mut path = fsmap.sym_to_path(&ent.name).await;
        path.push(OsStr::from_bytes(filename));
        if let Ok(meta) = path.symlink_metadata() {
            if meta.is_dir() {
                tokio::fs::remove_dir(&path).await.map_err(|_| nfsstat3::NFS3ERR_IO)?;
            } else {
                tokio::fs::remove_file(&path).await.map_err(|_| nfsstat3::NFS3ERR_IO)?;
            }

            let filesym = fsmap.intern.intern(OsStr::from_bytes(filename).to_os_string()).unwrap();
            let mut sympath = ent.name.clone();
            sympath.push(filesym);
            if let Some(fileid) = fsmap.path_to_id.get(&sympath).copied() {
                // update the fileid -> path
                // and the path -> fileid mappings for the deleted file
                fsmap.id_to_path.remove(&fileid);
                fsmap.path_to_id.remove(&sympath);
                // we need to update the children listing for the directories
                if let Ok(dirent_mut) = fsmap.find_entry_mut(dirid) {
                    if let Some(ref mut fromch) = dirent_mut.children {
                        fromch.remove(&fileid);
                    }
                }
            }

            let _ = fsmap.refresh_entry(dirid).await;
        } else {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }

        Ok(())
    }

    async fn rename(
        &self,
        from_dirid: fileid3,
        from_filename: &filename3,
        to_dirid: fileid3,
        to_filename: &filename3,
    ) -> Result<(), nfsstat3> {
        // Refuse rename if either endpoint touches the pollution namespace.
        // Source: NOENT (the name doesn't exist to us); destination: ACCES
        // (we won't let you create one of these via rename either).
        if is_pollution_pattern(from_filename) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        if is_pollution_pattern(to_filename) {
            return Err(nfsstat3::NFS3ERR_ACCES);
        }
        let mut fsmap = self.fsmap.lock().await;

        let from_dirent = fsmap.find_entry(from_dirid)?;
        let mut from_path = fsmap.sym_to_path(&from_dirent.name).await;
        from_path.push(OsStr::from_bytes(from_filename));

        let to_dirent = fsmap.find_entry(to_dirid)?;
        let mut to_path = fsmap.sym_to_path(&to_dirent.name).await;
        // to folder must exist
        if !exists_no_traverse(&to_path) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        to_path.push(OsStr::from_bytes(to_filename));

        // src path must exist
        if !exists_no_traverse(&from_path) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        debug!("Rename {:?} to {:?}", from_path, to_path);
        tokio::fs::rename(&from_path, &to_path)
            .await
            .map_err(|_| nfsstat3::NFS3ERR_IO)?;

        let oldsym = fsmap.intern.intern(OsStr::from_bytes(from_filename).to_os_string()).unwrap();
        let newsym = fsmap.intern.intern(OsStr::from_bytes(to_filename).to_os_string()).unwrap();

        let mut from_sympath = from_dirent.name.clone();
        from_sympath.push(oldsym);
        let mut to_sympath = to_dirent.name.clone();
        to_sympath.push(newsym);
        if let Some(fileid) = fsmap.path_to_id.get(&from_sympath).copied() {
            // update the fileid -> path
            // and the path -> fileid mappings for the new file
            fsmap.id_to_path.get_mut(&fileid).unwrap().name = to_sympath.clone();
            fsmap.path_to_id.remove(&from_sympath);
            fsmap.path_to_id.insert(to_sympath, fileid);
            if to_dirid != from_dirid {
                // moving across directories.
                // we need to update the children listing for the directories
                if let Ok(from_dirent_mut) = fsmap.find_entry_mut(from_dirid) {
                    if let Some(ref mut fromch) = from_dirent_mut.children {
                        fromch.remove(&fileid);
                    }
                }
                if let Ok(to_dirent_mut) = fsmap.find_entry_mut(to_dirid) {
                    if let Some(ref mut toch) = to_dirent_mut.children {
                        toch.insert(fileid);
                    }
                }
            }
        }
        let _ = fsmap.refresh_entry(from_dirid).await;
        if to_dirid != from_dirid {
            let _ = fsmap.refresh_entry(to_dirid).await;
        }

        Ok(())
    }
    async fn mkdir(&self, dirid: fileid3, dirname: &filename3) -> Result<(fileid3, fattr3), nfsstat3> {
        self.create_fs_object(dirid, dirname, &CreateFSObject::Directory).await
    }

    async fn symlink(
        &self,
        dirid: fileid3,
        linkname: &filename3,
        symlink: &nfspath3,
        attr: &sattr3,
    ) -> Result<(fileid3, fattr3), nfsstat3> {
        self.create_fs_object(dirid, linkname, &CreateFSObject::Symlink((*attr, symlink.clone())))
            .await
    }
    async fn readlink(&self, id: fileid3) -> Result<nfspath3, nfsstat3> {
        let fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(id)?;
        let path = fsmap.sym_to_path(&ent.name).await;
        drop(fsmap);
        if path.is_symlink() {
            if let Ok(target) = path.read_link() {
                Ok(target.as_os_str().as_bytes().into())
            } else {
                Err(nfsstat3::NFS3ERR_IO)
            }
        } else {
            Err(nfsstat3::NFS3ERR_BADTYPE)
        }
    }
}

/// CLI for `crb-nfsd`. Spawned by the claude-remote-bash daemon per export.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Bind address. Use `127.0.0.1:0` for an ephemeral port; the bound port
    /// is printed to stdout as `LISTEN_PORT=<n>` before the server enters its
    /// main loop so a parent process can capture it.
    #[arg(long, default_value = "127.0.0.1:0")]
    listen: String,

    /// Local directory to serve as the NFS root.
    #[arg(long)]
    root: PathBuf,

    /// Refuse all mutating NFS operations by reporting `VFSCapabilities::ReadOnly`.
    #[arg(long, default_value_t = false)]
    readonly: bool,

    /// Refuse to expose any path under this prefix (canonical absolute path).
    /// Repeatable. Used by the daemon to hide `$HOME/.crb/` (prevents recursive
    /// self-mount loops).
    #[arg(long, value_name = "PATH")]
    block_prefix: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use std::io::Write;

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    // Canonicalize so symlinks and relative paths don't slip past the block check.
    let root = args
        .root
        .canonicalize()
        .with_context(|| format!("--root {:?} could not be canonicalized", args.root))?;
    // Defensive metadata check. Upstream FSMap::new() does `root.metadata().unwrap()`
    // which would panic on a race (root removed between canonicalize and FSMap init).
    std::fs::metadata(&root).with_context(|| format!("--root {} metadata unreadable", root.display()))?;
    // Hard-fail if any --block-prefix can't be canonicalized — better to refuse
    // to start than to silently serve a path that should have been hidden.
    let block_prefixes: Vec<PathBuf> = args
        .block_prefix
        .iter()
        .map(|p| p.canonicalize().with_context(|| format!("--block-prefix {p:?} could not be canonicalized")))
        .collect::<anyhow::Result<_>>()?;

    let fs = MirrorFS::new(root, args.readonly, block_prefixes);
    let listener = NFSTcpListener::bind(&args.listen, fs).await?;

    // Print bound port + ready signal for the parent process. Flush immediately
    // — parent reads via a line-buffered pipe and expects these promptly.
    let port = listener.get_listen_port();
    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "LISTEN_PORT={port}")?;
    writeln!(stdout, "READY")?;
    stdout.flush()?;
    drop(stdout);

    listener.handle_forever().await.context("server exited")?;
    Ok(())
}
