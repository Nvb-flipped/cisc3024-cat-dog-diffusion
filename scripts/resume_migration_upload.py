"""Resume the migration uploader, adapting GitHub CLI's GraphQL release ID to REST.

Use this entry point with --upload-only after an interrupted upload. The snapshot
builder source is retained byte-identical to its already archived version.
"""
import json
import migration_snapshot


def rest_release_identity(command, output, call):
    if command[:2] != ('release', 'view') or '--json' not in command:
        return output
    fields = command[command.index('--json')+1].split(',')
    if 'id' not in fields:
        return output
    value = json.loads(output)
    if not isinstance(value.get('id'), str):
        return output
    repo = command[command.index('--repo')+1]
    tag = command[2]
    rest = json.loads(call('release','view',tag,'--repo',repo,'--json','apiUrl'))
    value['id'] = int(rest['apiUrl'].rstrip('/').rsplit('/',1)[-1])
    return json.dumps(value)


def main():
    original = migration_snapshot.gh
    def compatible(*command):
        return rest_release_identity(command, original(*command), original)
    migration_snapshot.gh = compatible
    migration_snapshot.main()


if __name__ == '__main__':
    main()
