"""Archive an explicit file list and verify bytes; no transcript discovery or text logging."""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode

import boto3
from botocore.exceptions import ClientError


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(data, name):
    result = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    if name.endswith('.csv'):
        reader = csv.reader(io.StringIO(data.decode()))
        result['columns'] = next(reader)
        result['rows'] = sum(1 for _ in reader)
    return result


def archive(s3, bucket, prefix, entries, root):
    objects = []
    for entry in entries:
        name = entry['id']
        data = (root / entry['local_path']).read_bytes()
        info = describe(data, name)
        key = f"{prefix.rstrip('/')}/sha256/{info['sha256']}/{Path(name).name}"
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=data, IfNoneMatch='*',
                          ServerSideEncryption='AES256',
                          ChecksumSHA256=base64.b64encode(hashlib.sha256(data).digest()).decode(),
                          Metadata={'sha256': info['sha256']},
                          Tagging=urlencode({'rr_project_id': 'probe-shift', 'rr_owner_id': 'aswin-surya',
                                             'rr_subproject_id': 'labels', 'rr_data_classification': 'internal'}))
        except ClientError as error:
            if error.response['Error']['Code'] not in ('PreconditionFailed', '412'):
                raise
        response = s3.get_object(Bucket=bucket, Key=key)
        downloaded = response['Body'].read()
        require(downloaded == data, 'Uploaded artifact verification failed')
        require(response.get('ServerSideEncryption') in ('AES256', 'aws:kms'), 'Missing encryption')
        objects.append({'id': name, **info, 'uri': f's3://{bucket}/{key}',
                        'version_id': response.get('VersionId'), 'verified_download_sha256': info['sha256'],
                        'role': entry['role']})
        print(json.dumps({'verified_objects': len(objects), 'bytes': info['bytes']}), flush=True)
    return objects


def fetch(s3, manifest, destination, only_ids=None):
    destination.mkdir(parents=True, exist_ok=True)
    if only_ids:
        require(set(only_ids) <= {item['id'] for item in manifest['objects']}, 'Unknown artifact ID')
    for item in manifest['objects']:
        if only_ids and item['id'] not in only_ids:
            continue
        rel = Path(item['id'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe artifact ID')
        bucket, key = item['uri'].removeprefix('s3://').split('/', 1)
        kwargs = {'Bucket': bucket, 'Key': key}
        if item.get('version_id'):
            kwargs['VersionId'] = item['version_id']
        data = s3.get_object(**kwargs)['Body'].read()
        require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], 'Downloaded artifact checksum mismatch')
        path = destination / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(path.read_bytes() == data, 'Refusing to overwrite a different local artifact')
        else:
            path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['archive', 'fetch'])
    parser.add_argument('--profile', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--account', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--bucket')
    parser.add_argument('--prefix')
    parser.add_argument('--source-commit')
    parser.add_argument('--id', action='append', dest='only_ids', help='Fetch only this exact manifest ID; repeatable')
    args = parser.parse_args()
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    identity = session.client('sts').get_caller_identity()
    require(identity['Account'] == args.account, 'Unexpected AWS account')
    require('/AWSReservedSSO_RedwoodResearcher_' in identity['Arn'], 'Unexpected AWS role')
    s3 = session.client('s3')
    if args.mode == 'fetch':
        fetch(s3, json.loads(args.manifest.read_text()), args.root, args.only_ids)
        return
    require(args.selection and args.bucket and args.prefix and args.source_commit, 'Missing archive arguments')
    require(not args.manifest.exists(), 'Never overwrite an artifact manifest')
    s3.head_bucket(Bucket=args.bucket, ExpectedBucketOwner=args.account)
    block = s3.get_public_access_block(Bucket=args.bucket)['PublicAccessBlockConfiguration']
    require(all(block[k] for k in ('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets')), 'Public access must be blocked')
    entries = json.loads(args.selection.read_text())
    require(len({e['id'] for e in entries}) == len(entries), 'Duplicate artifact ID')
    objects = archive(s3, args.bucket, args.prefix, entries, args.root)
    result = {'schema_version': 1, 'source_commit': args.source_commit, 'region': args.region,
              'account': args.account, 'public_access_block_verified': True, 'objects': objects}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
