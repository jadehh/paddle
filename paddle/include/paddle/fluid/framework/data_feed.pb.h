// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: data_feed.proto

#ifndef PROTOBUF_data_5ffeed_2eproto__INCLUDED
#define PROTOBUF_data_5ffeed_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3001000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3001000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace paddle {
namespace framework {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_data_5ffeed_2eproto();
void protobuf_InitDefaults_data_5ffeed_2eproto();
void protobuf_AssignDesc_data_5ffeed_2eproto();
void protobuf_ShutdownFile_data_5ffeed_2eproto();

class DataFeedDesc;
class MultiSlotDesc;
class Slot;

// ===================================================================

class Slot : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:paddle.framework.Slot) */ {
 public:
  Slot();
  virtual ~Slot();

  Slot(const Slot& from);

  inline Slot& operator=(const Slot& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Slot& default_instance();

  static const Slot* internal_default_instance();

  void Swap(Slot* other);

  // implements Message ----------------------------------------------

  inline Slot* New() const { return New(NULL); }

  Slot* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const Slot& from);
  void MergeFrom(const Slot& from);
  void Clear();
  bool IsInitialized() const;

  size_t ByteSizeLong() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(Slot* other);
  void UnsafeMergeFrom(const Slot& from);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required string name = 1;
  bool has_name() const;
  void clear_name();
  static const int kNameFieldNumber = 1;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);

  // required string type = 2;
  bool has_type() const;
  void clear_type();
  static const int kTypeFieldNumber = 2;
  const ::std::string& type() const;
  void set_type(const ::std::string& value);
  void set_type(const char* value);
  void set_type(const char* value, size_t size);
  ::std::string* mutable_type();
  ::std::string* release_type();
  void set_allocated_type(::std::string* type);

  // optional bool is_dense = 3 [default = false];
  bool has_is_dense() const;
  void clear_is_dense();
  static const int kIsDenseFieldNumber = 3;
  bool is_dense() const;
  void set_is_dense(bool value);

  // optional bool is_used = 4 [default = false];
  bool has_is_used() const;
  void clear_is_used();
  static const int kIsUsedFieldNumber = 4;
  bool is_used() const;
  void set_is_used(bool value);

  // repeated int32 shape = 5;
  int shape_size() const;
  void clear_shape();
  static const int kShapeFieldNumber = 5;
  ::google::protobuf::int32 shape(int index) const;
  void set_shape(int index, ::google::protobuf::int32 value);
  void add_shape(::google::protobuf::int32 value);
  const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
      shape() const;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
      mutable_shape();

  // @@protoc_insertion_point(class_scope:paddle.framework.Slot)
 private:
  inline void set_has_name();
  inline void clear_has_name();
  inline void set_has_type();
  inline void clear_has_type();
  inline void set_has_is_dense();
  inline void clear_has_is_dense();
  inline void set_has_is_used();
  inline void clear_has_is_used();

  // helper for ByteSizeLong()
  size_t RequiredFieldsByteSizeFallback() const;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< ::google::protobuf::int32 > shape_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::google::protobuf::internal::ArenaStringPtr type_;
  bool is_dense_;
  bool is_used_;
  friend void  protobuf_InitDefaults_data_5ffeed_2eproto_impl();
  friend void  protobuf_AddDesc_data_5ffeed_2eproto_impl();
  friend void protobuf_AssignDesc_data_5ffeed_2eproto();
  friend void protobuf_ShutdownFile_data_5ffeed_2eproto();

  void InitAsDefaultInstance();
};
extern ::google::protobuf::internal::ExplicitlyConstructed<Slot> Slot_default_instance_;

// -------------------------------------------------------------------

class MultiSlotDesc : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:paddle.framework.MultiSlotDesc) */ {
 public:
  MultiSlotDesc();
  virtual ~MultiSlotDesc();

  MultiSlotDesc(const MultiSlotDesc& from);

  inline MultiSlotDesc& operator=(const MultiSlotDesc& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const MultiSlotDesc& default_instance();

  static const MultiSlotDesc* internal_default_instance();

  void Swap(MultiSlotDesc* other);

  // implements Message ----------------------------------------------

  inline MultiSlotDesc* New() const { return New(NULL); }

  MultiSlotDesc* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const MultiSlotDesc& from);
  void MergeFrom(const MultiSlotDesc& from);
  void Clear();
  bool IsInitialized() const;

  size_t ByteSizeLong() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(MultiSlotDesc* other);
  void UnsafeMergeFrom(const MultiSlotDesc& from);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .paddle.framework.Slot slots = 1;
  int slots_size() const;
  void clear_slots();
  static const int kSlotsFieldNumber = 1;
  const ::paddle::framework::Slot& slots(int index) const;
  ::paddle::framework::Slot* mutable_slots(int index);
  ::paddle::framework::Slot* add_slots();
  ::google::protobuf::RepeatedPtrField< ::paddle::framework::Slot >*
      mutable_slots();
  const ::google::protobuf::RepeatedPtrField< ::paddle::framework::Slot >&
      slots() const;

  // @@protoc_insertion_point(class_scope:paddle.framework.MultiSlotDesc)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::RepeatedPtrField< ::paddle::framework::Slot > slots_;
  friend void  protobuf_InitDefaults_data_5ffeed_2eproto_impl();
  friend void  protobuf_AddDesc_data_5ffeed_2eproto_impl();
  friend void protobuf_AssignDesc_data_5ffeed_2eproto();
  friend void protobuf_ShutdownFile_data_5ffeed_2eproto();

  void InitAsDefaultInstance();
};
extern ::google::protobuf::internal::ExplicitlyConstructed<MultiSlotDesc> MultiSlotDesc_default_instance_;

// -------------------------------------------------------------------

class DataFeedDesc : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:paddle.framework.DataFeedDesc) */ {
 public:
  DataFeedDesc();
  virtual ~DataFeedDesc();

  DataFeedDesc(const DataFeedDesc& from);

  inline DataFeedDesc& operator=(const DataFeedDesc& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const DataFeedDesc& default_instance();

  static const DataFeedDesc* internal_default_instance();

  void Swap(DataFeedDesc* other);

  // implements Message ----------------------------------------------

  inline DataFeedDesc* New() const { return New(NULL); }

  DataFeedDesc* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const DataFeedDesc& from);
  void MergeFrom(const DataFeedDesc& from);
  void Clear();
  bool IsInitialized() const;

  size_t ByteSizeLong() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(DataFeedDesc* other);
  void UnsafeMergeFrom(const DataFeedDesc& from);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // optional string name = 1;
  bool has_name() const;
  void clear_name();
  static const int kNameFieldNumber = 1;
  const ::std::string& name() const;
  void set_name(const ::std::string& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  ::std::string* mutable_name();
  ::std::string* release_name();
  void set_allocated_name(::std::string* name);

  // optional int32 batch_size = 2 [default = 32];
  bool has_batch_size() const;
  void clear_batch_size();
  static const int kBatchSizeFieldNumber = 2;
  ::google::protobuf::int32 batch_size() const;
  void set_batch_size(::google::protobuf::int32 value);

  // optional .paddle.framework.MultiSlotDesc multi_slot_desc = 3;
  bool has_multi_slot_desc() const;
  void clear_multi_slot_desc();
  static const int kMultiSlotDescFieldNumber = 3;
  const ::paddle::framework::MultiSlotDesc& multi_slot_desc() const;
  ::paddle::framework::MultiSlotDesc* mutable_multi_slot_desc();
  ::paddle::framework::MultiSlotDesc* release_multi_slot_desc();
  void set_allocated_multi_slot_desc(::paddle::framework::MultiSlotDesc* multi_slot_desc);

  // optional string pipe_command = 4;
  bool has_pipe_command() const;
  void clear_pipe_command();
  static const int kPipeCommandFieldNumber = 4;
  const ::std::string& pipe_command() const;
  void set_pipe_command(const ::std::string& value);
  void set_pipe_command(const char* value);
  void set_pipe_command(const char* value, size_t size);
  ::std::string* mutable_pipe_command();
  ::std::string* release_pipe_command();
  void set_allocated_pipe_command(::std::string* pipe_command);

  // optional int32 thread_num = 5;
  bool has_thread_num() const;
  void clear_thread_num();
  static const int kThreadNumFieldNumber = 5;
  ::google::protobuf::int32 thread_num() const;
  void set_thread_num(::google::protobuf::int32 value);

  // @@protoc_insertion_point(class_scope:paddle.framework.DataFeedDesc)
 private:
  inline void set_has_name();
  inline void clear_has_name();
  inline void set_has_batch_size();
  inline void clear_has_batch_size();
  inline void set_has_multi_slot_desc();
  inline void clear_has_multi_slot_desc();
  inline void set_has_pipe_command();
  inline void clear_has_pipe_command();
  inline void set_has_thread_num();
  inline void clear_has_thread_num();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::internal::ArenaStringPtr name_;
  ::google::protobuf::internal::ArenaStringPtr pipe_command_;
  ::paddle::framework::MultiSlotDesc* multi_slot_desc_;
  ::google::protobuf::int32 thread_num_;
  ::google::protobuf::int32 batch_size_;
  friend void  protobuf_InitDefaults_data_5ffeed_2eproto_impl();
  friend void  protobuf_AddDesc_data_5ffeed_2eproto_impl();
  friend void protobuf_AssignDesc_data_5ffeed_2eproto();
  friend void protobuf_ShutdownFile_data_5ffeed_2eproto();

  void InitAsDefaultInstance();
};
extern ::google::protobuf::internal::ExplicitlyConstructed<DataFeedDesc> DataFeedDesc_default_instance_;

// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// Slot

// required string name = 1;
inline bool Slot::has_name() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Slot::set_has_name() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Slot::clear_has_name() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Slot::clear_name() {
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_name();
}
inline const ::std::string& Slot::name() const {
  // @@protoc_insertion_point(field_get:paddle.framework.Slot.name)
  return name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void Slot::set_name(const ::std::string& value) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:paddle.framework.Slot.name)
}
inline void Slot::set_name(const char* value) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:paddle.framework.Slot.name)
}
inline void Slot::set_name(const char* value, size_t size) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:paddle.framework.Slot.name)
}
inline ::std::string* Slot::mutable_name() {
  set_has_name();
  // @@protoc_insertion_point(field_mutable:paddle.framework.Slot.name)
  return name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* Slot::release_name() {
  // @@protoc_insertion_point(field_release:paddle.framework.Slot.name)
  clear_has_name();
  return name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void Slot::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    set_has_name();
  } else {
    clear_has_name();
  }
  name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:paddle.framework.Slot.name)
}

// required string type = 2;
inline bool Slot::has_type() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Slot::set_has_type() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Slot::clear_has_type() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Slot::clear_type() {
  type_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_type();
}
inline const ::std::string& Slot::type() const {
  // @@protoc_insertion_point(field_get:paddle.framework.Slot.type)
  return type_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void Slot::set_type(const ::std::string& value) {
  set_has_type();
  type_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:paddle.framework.Slot.type)
}
inline void Slot::set_type(const char* value) {
  set_has_type();
  type_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:paddle.framework.Slot.type)
}
inline void Slot::set_type(const char* value, size_t size) {
  set_has_type();
  type_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:paddle.framework.Slot.type)
}
inline ::std::string* Slot::mutable_type() {
  set_has_type();
  // @@protoc_insertion_point(field_mutable:paddle.framework.Slot.type)
  return type_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* Slot::release_type() {
  // @@protoc_insertion_point(field_release:paddle.framework.Slot.type)
  clear_has_type();
  return type_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void Slot::set_allocated_type(::std::string* type) {
  if (type != NULL) {
    set_has_type();
  } else {
    clear_has_type();
  }
  type_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), type);
  // @@protoc_insertion_point(field_set_allocated:paddle.framework.Slot.type)
}

// optional bool is_dense = 3 [default = false];
inline bool Slot::has_is_dense() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void Slot::set_has_is_dense() {
  _has_bits_[0] |= 0x00000004u;
}
inline void Slot::clear_has_is_dense() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void Slot::clear_is_dense() {
  is_dense_ = false;
  clear_has_is_dense();
}
inline bool Slot::is_dense() const {
  // @@protoc_insertion_point(field_get:paddle.framework.Slot.is_dense)
  return is_dense_;
}
inline void Slot::set_is_dense(bool value) {
  set_has_is_dense();
  is_dense_ = value;
  // @@protoc_insertion_point(field_set:paddle.framework.Slot.is_dense)
}

// optional bool is_used = 4 [default = false];
inline bool Slot::has_is_used() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void Slot::set_has_is_used() {
  _has_bits_[0] |= 0x00000008u;
}
inline void Slot::clear_has_is_used() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void Slot::clear_is_used() {
  is_used_ = false;
  clear_has_is_used();
}
inline bool Slot::is_used() const {
  // @@protoc_insertion_point(field_get:paddle.framework.Slot.is_used)
  return is_used_;
}
inline void Slot::set_is_used(bool value) {
  set_has_is_used();
  is_used_ = value;
  // @@protoc_insertion_point(field_set:paddle.framework.Slot.is_used)
}

// repeated int32 shape = 5;
inline int Slot::shape_size() const {
  return shape_.size();
}
inline void Slot::clear_shape() {
  shape_.Clear();
}
inline ::google::protobuf::int32 Slot::shape(int index) const {
  // @@protoc_insertion_point(field_get:paddle.framework.Slot.shape)
  return shape_.Get(index);
}
inline void Slot::set_shape(int index, ::google::protobuf::int32 value) {
  shape_.Set(index, value);
  // @@protoc_insertion_point(field_set:paddle.framework.Slot.shape)
}
inline void Slot::add_shape(::google::protobuf::int32 value) {
  shape_.Add(value);
  // @@protoc_insertion_point(field_add:paddle.framework.Slot.shape)
}
inline const ::google::protobuf::RepeatedField< ::google::protobuf::int32 >&
Slot::shape() const {
  // @@protoc_insertion_point(field_list:paddle.framework.Slot.shape)
  return shape_;
}
inline ::google::protobuf::RepeatedField< ::google::protobuf::int32 >*
Slot::mutable_shape() {
  // @@protoc_insertion_point(field_mutable_list:paddle.framework.Slot.shape)
  return &shape_;
}

inline const Slot* Slot::internal_default_instance() {
  return &Slot_default_instance_.get();
}
// -------------------------------------------------------------------

// MultiSlotDesc

// repeated .paddle.framework.Slot slots = 1;
inline int MultiSlotDesc::slots_size() const {
  return slots_.size();
}
inline void MultiSlotDesc::clear_slots() {
  slots_.Clear();
}
inline const ::paddle::framework::Slot& MultiSlotDesc::slots(int index) const {
  // @@protoc_insertion_point(field_get:paddle.framework.MultiSlotDesc.slots)
  return slots_.Get(index);
}
inline ::paddle::framework::Slot* MultiSlotDesc::mutable_slots(int index) {
  // @@protoc_insertion_point(field_mutable:paddle.framework.MultiSlotDesc.slots)
  return slots_.Mutable(index);
}
inline ::paddle::framework::Slot* MultiSlotDesc::add_slots() {
  // @@protoc_insertion_point(field_add:paddle.framework.MultiSlotDesc.slots)
  return slots_.Add();
}
inline ::google::protobuf::RepeatedPtrField< ::paddle::framework::Slot >*
MultiSlotDesc::mutable_slots() {
  // @@protoc_insertion_point(field_mutable_list:paddle.framework.MultiSlotDesc.slots)
  return &slots_;
}
inline const ::google::protobuf::RepeatedPtrField< ::paddle::framework::Slot >&
MultiSlotDesc::slots() const {
  // @@protoc_insertion_point(field_list:paddle.framework.MultiSlotDesc.slots)
  return slots_;
}

inline const MultiSlotDesc* MultiSlotDesc::internal_default_instance() {
  return &MultiSlotDesc_default_instance_.get();
}
// -------------------------------------------------------------------

// DataFeedDesc

// optional string name = 1;
inline bool DataFeedDesc::has_name() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void DataFeedDesc::set_has_name() {
  _has_bits_[0] |= 0x00000001u;
}
inline void DataFeedDesc::clear_has_name() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void DataFeedDesc::clear_name() {
  name_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_name();
}
inline const ::std::string& DataFeedDesc::name() const {
  // @@protoc_insertion_point(field_get:paddle.framework.DataFeedDesc.name)
  return name_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void DataFeedDesc::set_name(const ::std::string& value) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:paddle.framework.DataFeedDesc.name)
}
inline void DataFeedDesc::set_name(const char* value) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:paddle.framework.DataFeedDesc.name)
}
inline void DataFeedDesc::set_name(const char* value, size_t size) {
  set_has_name();
  name_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:paddle.framework.DataFeedDesc.name)
}
inline ::std::string* DataFeedDesc::mutable_name() {
  set_has_name();
  // @@protoc_insertion_point(field_mutable:paddle.framework.DataFeedDesc.name)
  return name_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* DataFeedDesc::release_name() {
  // @@protoc_insertion_point(field_release:paddle.framework.DataFeedDesc.name)
  clear_has_name();
  return name_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void DataFeedDesc::set_allocated_name(::std::string* name) {
  if (name != NULL) {
    set_has_name();
  } else {
    clear_has_name();
  }
  name_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:paddle.framework.DataFeedDesc.name)
}

// optional int32 batch_size = 2 [default = 32];
inline bool DataFeedDesc::has_batch_size() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void DataFeedDesc::set_has_batch_size() {
  _has_bits_[0] |= 0x00000002u;
}
inline void DataFeedDesc::clear_has_batch_size() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void DataFeedDesc::clear_batch_size() {
  batch_size_ = 32;
  clear_has_batch_size();
}
inline ::google::protobuf::int32 DataFeedDesc::batch_size() const {
  // @@protoc_insertion_point(field_get:paddle.framework.DataFeedDesc.batch_size)
  return batch_size_;
}
inline void DataFeedDesc::set_batch_size(::google::protobuf::int32 value) {
  set_has_batch_size();
  batch_size_ = value;
  // @@protoc_insertion_point(field_set:paddle.framework.DataFeedDesc.batch_size)
}

// optional .paddle.framework.MultiSlotDesc multi_slot_desc = 3;
inline bool DataFeedDesc::has_multi_slot_desc() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void DataFeedDesc::set_has_multi_slot_desc() {
  _has_bits_[0] |= 0x00000004u;
}
inline void DataFeedDesc::clear_has_multi_slot_desc() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void DataFeedDesc::clear_multi_slot_desc() {
  if (multi_slot_desc_ != NULL) multi_slot_desc_->::paddle::framework::MultiSlotDesc::Clear();
  clear_has_multi_slot_desc();
}
inline const ::paddle::framework::MultiSlotDesc& DataFeedDesc::multi_slot_desc() const {
  // @@protoc_insertion_point(field_get:paddle.framework.DataFeedDesc.multi_slot_desc)
  return multi_slot_desc_ != NULL ? *multi_slot_desc_
                         : *::paddle::framework::MultiSlotDesc::internal_default_instance();
}
inline ::paddle::framework::MultiSlotDesc* DataFeedDesc::mutable_multi_slot_desc() {
  set_has_multi_slot_desc();
  if (multi_slot_desc_ == NULL) {
    multi_slot_desc_ = new ::paddle::framework::MultiSlotDesc;
  }
  // @@protoc_insertion_point(field_mutable:paddle.framework.DataFeedDesc.multi_slot_desc)
  return multi_slot_desc_;
}
inline ::paddle::framework::MultiSlotDesc* DataFeedDesc::release_multi_slot_desc() {
  // @@protoc_insertion_point(field_release:paddle.framework.DataFeedDesc.multi_slot_desc)
  clear_has_multi_slot_desc();
  ::paddle::framework::MultiSlotDesc* temp = multi_slot_desc_;
  multi_slot_desc_ = NULL;
  return temp;
}
inline void DataFeedDesc::set_allocated_multi_slot_desc(::paddle::framework::MultiSlotDesc* multi_slot_desc) {
  delete multi_slot_desc_;
  multi_slot_desc_ = multi_slot_desc;
  if (multi_slot_desc) {
    set_has_multi_slot_desc();
  } else {
    clear_has_multi_slot_desc();
  }
  // @@protoc_insertion_point(field_set_allocated:paddle.framework.DataFeedDesc.multi_slot_desc)
}

// optional string pipe_command = 4;
inline bool DataFeedDesc::has_pipe_command() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void DataFeedDesc::set_has_pipe_command() {
  _has_bits_[0] |= 0x00000008u;
}
inline void DataFeedDesc::clear_has_pipe_command() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void DataFeedDesc::clear_pipe_command() {
  pipe_command_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  clear_has_pipe_command();
}
inline const ::std::string& DataFeedDesc::pipe_command() const {
  // @@protoc_insertion_point(field_get:paddle.framework.DataFeedDesc.pipe_command)
  return pipe_command_.GetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void DataFeedDesc::set_pipe_command(const ::std::string& value) {
  set_has_pipe_command();
  pipe_command_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:paddle.framework.DataFeedDesc.pipe_command)
}
inline void DataFeedDesc::set_pipe_command(const char* value) {
  set_has_pipe_command();
  pipe_command_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:paddle.framework.DataFeedDesc.pipe_command)
}
inline void DataFeedDesc::set_pipe_command(const char* value, size_t size) {
  set_has_pipe_command();
  pipe_command_.SetNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:paddle.framework.DataFeedDesc.pipe_command)
}
inline ::std::string* DataFeedDesc::mutable_pipe_command() {
  set_has_pipe_command();
  // @@protoc_insertion_point(field_mutable:paddle.framework.DataFeedDesc.pipe_command)
  return pipe_command_.MutableNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline ::std::string* DataFeedDesc::release_pipe_command() {
  // @@protoc_insertion_point(field_release:paddle.framework.DataFeedDesc.pipe_command)
  clear_has_pipe_command();
  return pipe_command_.ReleaseNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}
inline void DataFeedDesc::set_allocated_pipe_command(::std::string* pipe_command) {
  if (pipe_command != NULL) {
    set_has_pipe_command();
  } else {
    clear_has_pipe_command();
  }
  pipe_command_.SetAllocatedNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), pipe_command);
  // @@protoc_insertion_point(field_set_allocated:paddle.framework.DataFeedDesc.pipe_command)
}

// optional int32 thread_num = 5;
inline bool DataFeedDesc::has_thread_num() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void DataFeedDesc::set_has_thread_num() {
  _has_bits_[0] |= 0x00000010u;
}
inline void DataFeedDesc::clear_has_thread_num() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void DataFeedDesc::clear_thread_num() {
  thread_num_ = 0;
  clear_has_thread_num();
}
inline ::google::protobuf::int32 DataFeedDesc::thread_num() const {
  // @@protoc_insertion_point(field_get:paddle.framework.DataFeedDesc.thread_num)
  return thread_num_;
}
inline void DataFeedDesc::set_thread_num(::google::protobuf::int32 value) {
  set_has_thread_num();
  thread_num_ = value;
  // @@protoc_insertion_point(field_set:paddle.framework.DataFeedDesc.thread_num)
}

inline const DataFeedDesc* DataFeedDesc::internal_default_instance() {
  return &DataFeedDesc_default_instance_.get();
}
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace framework
}  // namespace paddle

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_data_5ffeed_2eproto__INCLUDED
