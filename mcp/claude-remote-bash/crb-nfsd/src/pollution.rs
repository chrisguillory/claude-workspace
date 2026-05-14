//! macOS pollution-name refusal and protocol-layer filename validation.
//!
//! Two pure helpers used as the trust boundary for every name that arrives
//! on the wire:
//!
//! - [`validate_component`] rejects empty, `.`, `..`, `/`-containing, and
//!   `\0`-containing names — closing path-traversal via filename injection.
//! - [`is_pollution_pattern`] filters macOS sidecar names (`.DS_Store`,
//!   `._*`, etc.) so opening the mount in Finder doesn't propagate Finder
//!   droppings to the peer's filesystem.
//!
//! Both are called from the NFS-protocol entry points in [`crate::fs`].

use nfsserve::nfs::nfsstat3;

/// Refuses any NFS `filename3` that is not a single, sane path component.
///
/// RFC 1813 §3.3.2 says `filename3` is one path component, but the upstream
/// `nfsserve` crate does not enforce that before delegating to the VFS impl.
/// Without this check, a peer crafting `filename = b"../../etc/whatever"`
/// would, on `PathBuf::push`, append those bytes literally; the OS resolves
/// `..` and `/` at syscall time, escaping `--root`. Enforce the contract at
/// the protocol layer so every mutation entry point fails loud and early.
pub(crate) fn validate_component(name: &[u8]) -> Result<(), nfsstat3> {
    if name.is_empty()
        || name == b"."
        || name == b".."
        || name.contains(&b'/')
        || name.contains(&b'\0')
    {
        return Err(nfsstat3::NFS3ERR_INVAL);
    }
    Ok(())
}

/// Refuses NFS symlink targets that escape the served tree at use-time.
///
/// `nfspath3` is opaque per RFC 1813 §3.3.17, but the kernel resolves the
/// stored string verbatim whenever something walks through the symlink.
/// Three classes of target turn an otherwise-benign symlink into a
/// confused-deputy primitive: absolute paths, `..` segments, and embedded
/// NULs. With the `read`/`write`/`setattr` type-gates already refusing
/// server-side opens through a NF3LNK fileid, this is defense-in-depth —
/// it also closes the self-mount case where a stock client's client-side
/// readlink resolves against the same filesystem the server sits on.
pub(crate) fn validate_symlink_target(target: &[u8]) -> Result<(), nfsstat3> {
    if target.is_empty()
        || target.first() == Some(&b'/')
        || target.contains(&b'\0')
        || target.split(|&b| b == b'/').any(|seg| seg == b"..")
    {
        return Err(nfsstat3::NFS3ERR_INVAL);
    }
    Ok(())
}

/// macOS metadata filenames that Finder, Spotlight, and Quick Look auto-create
/// on any browsed folder. Filtering these out of every namespace operation
/// (lookup, readdir, create, remove, rename) keeps Finder/Spotlight from
/// polluting the peer's filesystem when the mount is opened in GUI.
pub(crate) fn is_pollution_pattern(name: &[u8]) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pollution_pattern_matches_exact_macos_names() {
        assert!(is_pollution_pattern(b".DS_Store"));
        assert!(is_pollution_pattern(b".Spotlight-V100"));
        assert!(is_pollution_pattern(b".Trashes"));
        assert!(is_pollution_pattern(b".fseventsd"));
        assert!(is_pollution_pattern(b".TemporaryItems"));
        assert!(is_pollution_pattern(b".DocumentRevisions-V100"));
    }

    #[test]
    fn pollution_pattern_matches_icon_with_literal_cr() {
        // Finder's "Custom Folder Icon" file is literally [0x49, 0x63, 0x6f, 0x6e, 0x0D].
        // Both spellings must match the same 5-byte sequence.
        assert!(is_pollution_pattern(b"Icon\r"));
        assert!(is_pollution_pattern(&[0x49, 0x63, 0x6f, 0x6e, 0x0D]));
    }

    #[test]
    fn pollution_pattern_matches_appledouble_prefix() {
        assert!(is_pollution_pattern(b"._foo"));
        assert!(is_pollution_pattern(b"._.DS_Store"));
        assert!(is_pollution_pattern(b"._main.rs"));
        // The two-byte minimum prefix matches on its own.
        assert!(is_pollution_pattern(b"._"));
    }

    #[test]
    fn pollution_pattern_rejects_regular_names() {
        assert!(!is_pollution_pattern(b"regular.txt"));
        assert!(!is_pollution_pattern(b"main.rs"));
        assert!(!is_pollution_pattern(b".gitignore"));
        assert!(!is_pollution_pattern(b".env"));
        assert!(!is_pollution_pattern(b""));
    }

    #[test]
    fn pollution_pattern_is_case_sensitive() {
        // NFS LOOKUP is byte-exact; macOS writers use these exact capitalizations.
        assert!(!is_pollution_pattern(b".ds_store"));
        assert!(!is_pollution_pattern(b".DS_STORE"));
        assert!(!is_pollution_pattern(b"icon\r"));
    }

    #[test]
    fn pollution_pattern_rejects_icon_without_cr() {
        // Plain "Icon" is a valid user filename; only the CR-suffixed Finder pattern is refused.
        assert!(!is_pollution_pattern(b"Icon"));
        assert!(!is_pollution_pattern(b"Icon\n"));
        assert!(!is_pollution_pattern(b"Icon\r\n"));
    }

    #[test]
    fn pollution_pattern_dot_underscore_is_prefix_only() {
        // "._" matches as a prefix, not as a substring elsewhere in the name.
        assert!(!is_pollution_pattern(b"foo._bar"));
        assert!(!is_pollution_pattern(b" ._"));
    }

    #[test]
    fn validate_component_rejects_path_traversal() {
        assert!(validate_component(b"..").is_err());
        assert!(validate_component(b"../etc/passwd").is_err());
        assert!(validate_component(b"foo/bar").is_err());
        assert!(validate_component(b"/absolute").is_err());
        assert!(validate_component(b"../../tmp/pwned").is_err());
    }

    #[test]
    fn validate_component_rejects_special_names() {
        assert!(validate_component(b"").is_err());
        assert!(validate_component(b".").is_err());
    }

    #[test]
    fn validate_component_rejects_null_byte() {
        assert!(validate_component(b"foo\0bar").is_err());
        assert!(validate_component(b"\0").is_err());
    }

    #[test]
    fn validate_symlink_target_rejects_absolute() {
        assert!(validate_symlink_target(b"/etc/passwd").is_err());
        assert!(validate_symlink_target(b"/").is_err());
        assert!(validate_symlink_target(b"/Users/chris/.ssh/authorized_keys").is_err());
    }

    #[test]
    fn validate_symlink_target_rejects_parent_segment() {
        assert!(validate_symlink_target(b"..").is_err());
        assert!(validate_symlink_target(b"../etc").is_err());
        assert!(validate_symlink_target(b"foo/../../etc").is_err());
        assert!(validate_symlink_target(b"a/b/c/../../..").is_err());
    }

    #[test]
    fn validate_symlink_target_rejects_null_and_empty() {
        assert!(validate_symlink_target(b"").is_err());
        assert!(validate_symlink_target(b"foo\0bar").is_err());
    }

    #[test]
    fn validate_symlink_target_accepts_relative_paths() {
        assert!(validate_symlink_target(b"sibling.txt").is_ok());
        assert!(validate_symlink_target(b"subdir/child").is_ok());
        assert!(validate_symlink_target(b"a/b/c/d").is_ok());
        // Names containing ".." as a substring are fine; only `..` as a full
        // path segment is dangerous.
        assert!(validate_symlink_target(b"file..txt").is_ok());
        assert!(validate_symlink_target(b"...").is_ok());
    }

    #[test]
    fn validate_component_accepts_normal_filenames() {
        assert!(validate_component(b"main.rs").is_ok());
        assert!(validate_component(b".gitignore").is_ok());
        assert!(validate_component(b".hidden").is_ok());
        assert!(validate_component(b"a").is_ok());
        // Dot-prefixed name that is NOT just "." or ".." is fine.
        assert!(validate_component(b"...").is_ok());
        // Spaces and unicode in single component are fine.
        assert!(validate_component(b"file with spaces.txt").is_ok());
    }
}
