//! NFSv3 mirror filesystem.
//!
//! [`MirrorFS`] implements [`NFSFileSystem`] by mirroring a local directory.
//! [`FSMap`] is an internal cache mapping NFS fileids to filesystem entries.
//! The pollution and validation policy lives in [`crate::pollution`].

use std::collections::{BTreeSet, HashMap};
use std::ffi::{OsStr, OsString};
use std::fs::Metadata;
use std::io::SeekFrom;
use std::ops::Bound;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use intaglio::osstr::SymbolTable;
use intaglio::Symbol;
use nfsserve::fs_util::{
    exists_no_traverse, fattr3_differ, file_setattr, metadata_to_fattr3, path_setattr,
};
use nfsserve::nfs::{fattr3, fileid3, filename3, ftype3, nfspath3, nfsstat3, sattr3};
use nfsserve::vfs::{DirEntry, NFSFileSystem, ReadDirResult, VFSCapabilities};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tracing::debug;

use crate::pollution::{is_pollution_pattern, validate_component, validate_symlink_target};

/// Matches the `rtmax` advertised in nfsserve's `fsinfo` response. READ
/// requests beyond this cap are refused with `NFS3ERR_INVAL` — the server
/// won't allocate a multi-GB response buffer just because a peer asked.
const MAX_READ_SIZE: usize = 1024 * 1024;

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
    fn sym_to_path(&self, symlist: &[Symbol]) -> PathBuf {
        let mut ret = self.root.clone();
        for i in symlist.iter() {
            ret.push(self.intern.get(*i).unwrap());
        }
        ret
    }

    fn sym_to_fname(&self, symlist: &[Symbol]) -> OsString {
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
    fn find_child(&self, id: fileid3, filename: &[u8]) -> Result<fileid3, nfsstat3> {
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
        let path = self.sym_to_path(&entry.name);
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
        let path = self.sym_to_path(&entry.name);
        let mut new_children: Vec<u64> = Vec::new();
        debug!("Relisting entry {:?}: {:?}. Ent: {:?}", id, path, entry);
        if let Ok(mut listing) = tokio::fs::read_dir(&path).await {
            while let Some(entry) = listing.next_entry().await.map_err(|_| nfsstat3::NFS3ERR_IO)? {
                let sym = self
                    .intern
                    .intern(entry.file_name())
                    .map_err(|_| nfsstat3::NFS3ERR_IO)?;
                cur_path.push(sym);
                // symlink_metadata (not entry.metadata()) so a symlinked child is
                // cached as NF3LNK with the link's own attrs — never as the
                // target's attrs. Without this, a symlink-to-blocked-target would
                // be served as a regular file by the cache, and read() would
                // follow it server-side, bypassing --block-prefix.
                let meta = tokio::fs::symlink_metadata(entry.path())
                    .await
                    .map_err(|_| nfsstat3::NFS3ERR_IO)?;
                let next_id = self.create_entry(&cur_path, meta);
                new_children.push(next_id);
                cur_path.pop();
            }
            self.id_to_path.get_mut(&id).ok_or(nfsstat3::NFS3ERR_NOENT)?.children =
                Some(BTreeSet::from_iter(new_children.into_iter()));
        }

        Ok(())
    }

    fn create_entry(&mut self, fullpath: &[Symbol], meta: Metadata) -> fileid3 {
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
pub(crate) struct MirrorFS {
    fsmap: tokio::sync::Mutex<FSMap>,
    readonly: bool,
    /// Canonical absolute path prefixes that must not be exposed. Hit in `lookup`.
    block_prefixes: Box<[PathBuf]>,
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
    pub(crate) fn new(root: PathBuf, readonly: bool, block_prefixes: Box<[PathBuf]>) -> MirrorFS {
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
        validate_component(objectname)?;
        // Refuse Apple metadata sidecar names so Finder/Spotlight/Quick Look
        // can't pollute the peer's filesystem with `.DS_Store`, `._*`, etc.
        // EACCES (not EPERM) — per RFC 1813 EACCES is the policy-refused status
        // Finder handles gracefully; EPERM can trigger elevated-retry paths.
        if is_pollution_pattern(objectname) {
            return Err(nfsstat3::NFS3ERR_ACCES);
        }
        let mut fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(dirid)?;
        // dirid must resolve to a directory. Without this, a peer holding a
        // fileid for a peer-created symlink could pass it as dirid; the
        // sym_to_path + push below produces a path the OS walks through the
        // symlink at create-time, escaping --root. Same shape applies to
        // mkdir, symlink, exclusive-create — all funnel through here.
        if !matches!(ent.fsmeta.ftype, ftype3::NF3DIR) {
            return Err(nfsstat3::NFS3ERR_NOTDIR);
        }
        let mut path = fsmap.sym_to_path(&ent.name);
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
                // O_NOFOLLOW: refuse to create through a peer-planted symlink
                // at the leaf. Plain File::create is O_WRONLY|O_CREAT|O_TRUNC
                // without O_NOFOLLOW, which the kernel resolves through a
                // leaf symlink — truncating whatever the daemon's uid can
                // write at the target (e.g., ~/.ssh/authorized_keys).
                let file = std::fs::OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .custom_flags(libc::O_NOFOLLOW)
                    .open(&path)
                    .map_err(|e| match e.raw_os_error() {
                        Some(libc::ELOOP) => nfsstat3::NFS3ERR_INVAL,
                        _ => nfsstat3::NFS3ERR_IO,
                    })?;
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
                validate_symlink_target(target)?;
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
        let fileid = fsmap.create_entry(&name, meta.clone());

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
        validate_component(filename)?;
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
        // dirid must be a directory. Defense in depth — refresh_dir_list's
        // early-return on non-NF3DIR (line 164) already prevents lookup from
        // populating non-dir children, but a peer-held symlink fileid could
        // still flow through the path-build path below; fail loud at the
        // protocol layer to be consistent with create/remove/rename.
        if !matches!(dirent.fsmeta.ftype, ftype3::NF3DIR) {
            return Err(nfsstat3::NFS3ERR_NOTDIR);
        }
        let mut path = fsmap.sym_to_path(&dirent.name);
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

        if let Ok(id) = fsmap.find_child(dirid, filename) {
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

        fsmap.find_child(dirid, filename)
    }

    async fn getattr(&self, id: fileid3) -> Result<fattr3, nfsstat3> {
        let mut fsmap = self.fsmap.lock().await;
        if let RefreshResult::Delete = fsmap.refresh_entry(id).await? {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        let ent = fsmap.find_entry(id)?;
        let path = fsmap.sym_to_path(&ent.name);
        debug!("Stat {:?}: {:?}", path, ent);
        Ok(ent.fsmeta)
    }

    async fn read(&self, id: fileid3, offset: u64, count: u32) -> Result<(Vec<u8>, bool), nfsstat3> {
        // Enforce the rtmax we advertise in fsinfo (1 MB). Without this, a
        // peer-controlled `count` up to u32::MAX would drive a multi-GB
        // `vec![0; count]` allocation per RPC — recoverable DoS via the
        // server's own response buffer. RFC 1813 §3.3.6 permits servers to
        // return short reads, so capping is RFC-compliant.
        if count as usize > MAX_READ_SIZE {
            return Err(nfsstat3::NFS3ERR_INVAL);
        }
        let fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(id)?;
        // RFC 1813 §3.3.6: READ is defined for regular files only. Type-gate
        // before the open call — File::open follows symlinks at the final
        // component, so without this check a peer-created symlink in the
        // served tree (target = anywhere outside --root, e.g. /etc/passwd or
        // a path under --block-prefix) would be readable via this handler.
        // refresh_dir_list caches symlinks as NF3LNK in their fattr3; this
        // gate enforces the type at the protocol layer.
        if !matches!(ent.fsmeta.ftype, ftype3::NF3REG) {
            return Err(nfsstat3::NFS3ERR_INVAL);
        }
        let path = fsmap.sym_to_path(&ent.name);
        drop(fsmap);
        // O_NOFOLLOW closes a TOCTOU between the type-gate above and this
        // open: peer A drops the lock here, peer B runs remove(name) then
        // symlink(name, "<relative-target>"), peer A's open follows the
        // freshly-planted symlink and bypasses the gate. The type-gate's
        // cache check was valid at lock time but the leaf inode can change
        // after lock release. ELOOP -> INVAL surfaces the refusal explicitly.
        let mut f = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)
            .await
            .map_err(|e| {
                debug!("read: open({:?}) failed: {}", path, e);
                match e.raw_os_error() {
                    Some(err) if err == libc::ELOOP => nfsstat3::NFS3ERR_INVAL,
                    _ => match e.kind() {
                        std::io::ErrorKind::NotFound => nfsstat3::NFS3ERR_NOENT,
                        std::io::ErrorKind::PermissionDenied => nfsstat3::NFS3ERR_ACCES,
                        std::io::ErrorKind::IsADirectory => nfsstat3::NFS3ERR_ISDIR,
                        _ => nfsstat3::NFS3ERR_IO,
                    },
                }
            })?;
        let len = f.metadata().await.map_err(|e| {
            debug!("read: metadata({:?}) failed: {}", path, e);
            nfsstat3::NFS3ERR_IO
        })?.len();
        let mut start = offset;
        // checked_add: real NFSv3 clients cap `count` at ~64KB so overflow is
        // theoretical, but a malformed tunnel-claim could craft `offset + count`
        // to wrap past u64::MAX. Refuse explicitly.
        let mut end = offset
            .checked_add(u64::from(count))
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
        let path = fsmap.sym_to_path(&entry.name);
        debug!("path: {:?}", path);
        debug!("children len: {:?}", children.len());
        debug!("remaining_len : {:?}", remaining_length);
        // skipped == hidden-from-client entries (blocked-prefix or pollution-pattern).
        // End-of-stream is reached when entries_returned + skipped == remaining_length.
        let mut skipped = 0_usize;
        for i in children.range((range_start, Bound::Unbounded)) {
            let fileid = *i;
            let fileent = fsmap.find_entry(fileid)?;
            let name = fsmap.sym_to_fname(&fileent.name);
            if is_pollution_pattern(name.as_bytes()) {
                skipped += 1;
                continue;
            }
            let child_path = fsmap.sym_to_path(&fileent.name);
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
        // Refuse SETATTR on symlinks. path_setattr uses filetime::set_file_*
        // and std::fs::set_permissions, both of which follow symlinks at the
        // final component on macOS. A peer-created symlink whose target lives
        // outside --root could otherwise be used to truncate (size=0), chmod,
        // or touch any file the daemon's user can write. lchmod/lutimes
        // semantics aren't worth surfacing for symlinks themselves; refuse.
        // Dirs remain allowed (legit atime/mtime/mode use).
        if matches!(entry.fsmeta.ftype, ftype3::NF3LNK) {
            return Err(nfsstat3::NFS3ERR_INVAL);
        }
        let path = fsmap.sym_to_path(&entry.name);
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
        // RFC 1813 §3.3.7: WRITE is for regular files only. Same threat as
        // read() — OpenOptions::open follows symlinks at the final component,
        // and with create(true) a peer-crafted symlink to a non-existent
        // target could even synthesize a new file at an arbitrary path
        // (e.g. ~/.ssh/authorized_keys, /etc/cron.d/payload). Refuse non-REG
        // at the protocol layer.
        if !matches!(ent.fsmeta.ftype, ftype3::NF3REG) {
            return Err(nfsstat3::NFS3ERR_INVAL);
        }
        let path = fsmap.sym_to_path(&ent.name);
        drop(fsmap);
        debug!("write to init {:?}", path);
        // O_NOFOLLOW: same TOCTOU defense as read() — close the race where a
        // peer can substitute the leaf with a symlink between our cache
        // type-check and this open. Without O_NOFOLLOW, create(true) on a
        // dangling symlink would also synthesize a file at the resolved
        // target.
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&path)
            .await
            .map_err(|e| {
                debug!("write: open({:?}) failed: {}", path, e);
                match e.raw_os_error() {
                    Some(err) if err == libc::ELOOP => nfsstat3::NFS3ERR_INVAL,
                    _ => nfsstat3::NFS3ERR_IO,
                }
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
        validate_component(filename)?;
        // Apple pollution sidecars are namespace-invisible — report NOENT,
        // consistent with lookup() and readdir() filtering.
        if is_pollution_pattern(filename) {
            return Err(nfsstat3::NFS3ERR_NOENT);
        }
        let mut fsmap = self.fsmap.lock().await;
        let ent = fsmap.find_entry(dirid)?;
        // dirid must be a directory. Without this, remove on a peer-created
        // symlink-as-parent would unlink files outside --root.
        if !matches!(ent.fsmeta.ftype, ftype3::NF3DIR) {
            return Err(nfsstat3::NFS3ERR_NOTDIR);
        }
        let mut path = fsmap.sym_to_path(&ent.name);
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
        validate_component(from_filename)?;
        validate_component(to_filename)?;
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
        // Both dirids must be directories. Same threat as create/remove —
        // a symlinked-dir endpoint would let rename move files out of --root.
        if !matches!(from_dirent.fsmeta.ftype, ftype3::NF3DIR) {
            return Err(nfsstat3::NFS3ERR_NOTDIR);
        }
        let mut from_path = fsmap.sym_to_path(&from_dirent.name);
        from_path.push(OsStr::from_bytes(from_filename));

        let to_dirent = fsmap.find_entry(to_dirid)?;
        if !matches!(to_dirent.fsmeta.ftype, ftype3::NF3DIR) {
            return Err(nfsstat3::NFS3ERR_NOTDIR);
        }
        let mut to_path = fsmap.sym_to_path(&to_dirent.name);
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
        tokio::fs::rename(&from_path, &to_path).await.map_err(|e| {
            debug!("rename({:?} -> {:?}) failed: {}", from_path, to_path, e);
            // Preserve RFC 1813 RENAME3resfail semantics so the Edit-tool's
            // atomic write-temp-then-rename flow can distinguish "target was a
            // non-empty dir" from "permission denied" from "cross-filesystem".
            match e.kind() {
                std::io::ErrorKind::NotFound => nfsstat3::NFS3ERR_NOENT,
                std::io::ErrorKind::PermissionDenied => nfsstat3::NFS3ERR_ACCES,
                std::io::ErrorKind::AlreadyExists => nfsstat3::NFS3ERR_EXIST,
                std::io::ErrorKind::IsADirectory => nfsstat3::NFS3ERR_ISDIR,
                std::io::ErrorKind::NotADirectory => nfsstat3::NFS3ERR_NOTDIR,
                std::io::ErrorKind::DirectoryNotEmpty => nfsstat3::NFS3ERR_NOTEMPTY,
                std::io::ErrorKind::CrossesDevices => nfsstat3::NFS3ERR_XDEV,
                _ => nfsstat3::NFS3ERR_IO,
            }
        })?;

        let oldsym = fsmap.intern.intern(OsStr::from_bytes(from_filename).to_os_string()).unwrap();
        let newsym = fsmap.intern.intern(OsStr::from_bytes(to_filename).to_os_string()).unwrap();

        let mut from_sympath = from_dirent.name.clone();
        from_sympath.push(oldsym);
        let mut to_sympath = to_dirent.name.clone();
        to_sympath.push(newsym);
        if let Some(fileid) = fsmap.path_to_id.get(&from_sympath).copied() {
            // Evict any pre-existing destination entry first. Without this,
            // an overwrite leaves the displaced fileid's id_to_path entry
            // intact with its old (now-stale) ftype/name; a peer holding the
            // stale handle can then READ/WRITE through it, defeating the
            // ftype gate at read()/write() since the cached entry still
            // says NF3REG while the on-disk leaf is the renamed symlink.
            if let Some(stale_id) = fsmap.path_to_id.get(&to_sympath).copied() {
                if stale_id != fileid {
                    if to_dirid != from_dirid {
                        if let Ok(to_dirent_mut) = fsmap.find_entry_mut(to_dirid) {
                            if let Some(ref mut toch) = to_dirent_mut.children {
                                toch.remove(&stale_id);
                            }
                        }
                    }
                    fsmap.delete_entry(stale_id);
                }
            }
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
        let path = fsmap.sym_to_path(&ent.name);
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
